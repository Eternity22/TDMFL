import torch
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
import torch.nn as nn
from torchvision.models import resnet18
import matplotlib.pyplot
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from config.config import cfg
from scheduler import create_scheduler

#
model=resnet18(pretrained=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

scheduler = create_scheduler(cfg, optimizer)
plt.figure()
max_epoch=30
iters=20
cur_lr_list = []
lr_list = [4e-6,2.02e-4,3.73e-4,3.42e-4,3e-4,2.5e-4,2e-4,1.49e-4,1.02e-4,6.03e-5,2.87e-5,8.78e-6]
epochs =1
EPOCH = 26
for epoch in range(epochs, EPOCH+1):
    print(epoch)

    scheduler.step(epoch)

    cur_lr=optimizer.param_groups[-1]['lr']
    cur_lr_list.append(cur_lr)
    print('cur_lr:',cur_lr)
x_list = list(range(len(cur_lr_list)))
plt.plot(x_list, cur_lr_list)
plt.show()