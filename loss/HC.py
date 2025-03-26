import random

import numpy as np
from torch import nn, tensor
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class hetero_loss(nn.Module):
    def __init__(self, margin=0.1, dist_type='l2'):

        super(hetero_loss, self).__init__()
        self.feat_norm = "yes"

        self.margin = margin
        self.dist_type = dist_type
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()

    def forward(self, feat1, feat2, label1, label2):
        feat1 = F.normalize(feat1, p=2, dim=-1)
        feat2 = F.normalize(feat2, p=2, dim=-1)
        feat_size = feat1.size()[1]
        feat_num = feat1.size()[0]
        label_num = len(label1.unique())
        feat1 = feat1.chunk(label_num, 0)
        feat2 = feat2.chunk(label_num, 0)
        # loss = Variable(.cuda())
        for i in range(label_num):
            center1 = torch.mean(feat1[i], dim=0)
            center2 = torch.mean(feat2[i], dim=0)
            if self.dist_type == 'l2' or self.dist_type == 'l1':
                if i == 0:
                    dist = max(0.0, self.dist(center1, center2) - self.margin)
                else:
                    dist += max(0.0, self.dist(center1, center2) - self.margin)
            elif self.dist_type == 'cos':
                if i == 0:
                    dist = max(0.0, 1 - self.dist(center1, center2) - self.margin)
                else:
                    dist += max(0.0, 1 - self.dist(center1, center2) - self.margin)

        return dist

class hetero_loss_1(nn.Module):
    def __init__(self, margin=0.1, dist_type='l2'):

        super(hetero_loss_1, self).__init__()
        self.feat_norm = "yes"

        self.margin = margin
        self.dist_type = dist_type
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()

    def forward(self, feat1, feat2, label1, label2):
        feat1 = F.normalize(feat1, p=2, dim=-1)
        feat2 = F.normalize(feat2, p=2, dim=-1)
        feat_size = feat1.size()[1]
        feat_num = feat1.size()[0]
        label_num = len(label1.unique())
        feat1 = feat1.chunk(label_num, 0)

        feat2 = feat2.chunk(label_num, 0)
        # loss = Variable(.cuda())
        for i in range(label_num):
            # 随机选取2个样本组成中心
            index1,index2,index3,index4 = random.randint(0,3),random.randint(0,3),random.randint(0,3),random.randint(0,3)
            center1 = torch.mean((feat1[i][index1],feat1[i][index2]), dim=0)
            center2 = torch.mean((feat2[i][index3],feat2[i][index4]), dim=0)
            if self.dist_type == 'l2' or self.dist_type == 'l1':
                if i == 0:
                    dist = max(0, self.dist(center1, center2) - self.margin)
                else:
                    dist += max(0, self.dist(center1, center2) - self.margin)
            elif self.dist_type == 'cos':
                if i == 0:
                    dist = max(0, 1 - self.dist(center1, center2) - self.margin)
                else:
                    dist += max(0, 1 - self.dist(center1, center2) - self.margin)

        return dist