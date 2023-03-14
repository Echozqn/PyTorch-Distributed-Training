import argparse
import time
from datetime import datetime

import torch
import torchvision
from torch import distributed as dist
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)  # 与其他worker进行同步
        if rank == 0:
            tensor /= world_size


parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, help="local gpu id")
parser.add_argument('--batch_size', default=128, type=int, help="batch size")

args = parser.parse_args()
batch_size = args.batch_size
epochs = 2
lr = 0.001

dist.init_process_group(backend='nccl', init_method='env://')
torch.cuda.set_device(args.local_rank)
global_rank = dist.get_rank()
world_size = dist.get_world_size()


class ResNetMNIST(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(num_classes=10)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        return self.model(x)


net = ResNetMNIST()

data_root = 'dataset'
trainset = MNIST(root=data_root,
                 download=True,
                 train=True,
                 transform=ToTensor())

valset = MNIST(root=data_root,
               download=True,
               train=False,
               transform=ToTensor())

sampler = DistributedSampler(trainset)
train_loader = DataLoader(trainset,
                          batch_size=batch_size,
                          shuffle=False,
                          pin_memory=True,
                          sampler=sampler)

val_loader = DataLoader(valset,
                        batch_size=batch_size,
                        shuffle=False,
                        pin_memory=True)

file_name = f"{batch_size}_{global_rank}.log"
data_file = open(file_name, "w")
data_file.write("datetime\tg_step\tg_img\tloss_value\texamples_per_sec\n")
import torch.profiler

with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name='./zqn'),
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True
) as p:
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.cuda()
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank)
    net.train()
    print("Start")
    global_step = 0
    train_begin = time.time()
    for e in range(epochs):
        # DistributedSampler deterministically shuffle data
        # by seting random seed be current number epoch
        # so if do not call set_epoch when start of one epoch
        # the order of shuffled data will be always same
        sampler.set_epoch(e)
        for idx, (imgs, labels) in enumerate(train_loader):
            start = time.time()

            global_step += 1
            imgs = imgs.cuda()  # loading
            labels = labels.cuda()  # loading
            output = net(imgs)  # running
            loss = criterion(output, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            reduce_loss(loss, global_rank, world_size)

            if global_rank == 0 and global_step % 5 == 0:
                duration = time.time() - start
                examples_per_sec = batch_size / duration
                val = f"{datetime.now()}\t{global_step * world_size}\t{global_step * world_size * batch_size}\t{loss.item()}\t{examples_per_sec}\n"
                data_file.write(val)

            if idx % 10 == 0 and global_rank == 0:
                print('Epoch: {} step: {} loss: {}'.format(e, idx, loss.item()))
            p.step()

        data_file.write("TrainTime\t%f\n" % (time.time() - train_begin))

    # net.eval()
    # with torch.no_grad():
    #     cnt = 0
    #     total = len(val_loader.dataset)
    #     for imgs, labels in val_loader:
    #         imgs, labels = imgs.cuda(), labels.cuda()
    #         output = net(imgs)
    #         predict = torch.argmax(output, dim=1)
    #         cnt += (predict == labels).sum().item()

    # if global_rank == 0:
    #     print('eval accuracy: {}'.format(cnt / total))
