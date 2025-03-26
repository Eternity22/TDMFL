import torch
from torch import nn
import torch.nn.functional as F
import random
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx


class DCL(nn.Module):
    def __init__(self, num_pos=4, feat_norm='no'):
        super(DCL, self).__init__()
        self.num_pos = num_pos
        self.feat_norm = feat_norm

    def forward(self,inputs, targets):
        if self.feat_norm == 'yes':
            inputs = F.normalize(inputs, p=2, dim=-1)

        N = inputs.size(0)
        id_num = N // 2 // self.num_pos

        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())
        is_neg_c2i = is_neg[::self.num_pos, :].chunk(2, 0)[0]  # mask [id_num, N]

        centers = []
        for i in range(id_num):
            centers.append(inputs[targets == targets[i * self.num_pos]].mean(0))
        centers = torch.stack(centers)

        dist_mat = pdist_torch(centers, inputs)  #  c-i

        an = dist_mat * is_neg_c2i
        an = an[an > 1e-6].view(id_num, -1)

        # d_neg = torch.mean(an, dim=1, keepdim=True)
        # mask_an = (an - d_neg).expand(id_num, N - 2 * self.num_pos).lt(0)  # mask
        # an = an * mask_an
        #
        # list_an = []
        # for i in range (id_num):
        #     list_an.append(torch.mean(an[i][an[i]>1e-6]))
       #an_mean = sum(list_an) / len(list_an)
        an_mean = torch.mean(an)
        ap = dist_mat * ~is_neg_c2i
        ap_mean = torch.mean(ap[ap>1e-6])

        loss = ap_mean / an_mean
        return loss



# class DCL(nn.Module):
#     def __init__(self, num_pos=4, feat_norm='no'):
#         super(DCL, self).__init__()
#         self.num_pos = num_pos
#         self.feat_norm = feat_norm
#
#     def forward(self,inputs, targets):
#         if self.feat_norm == 'yes':
#             inputs = F.normalize(inputs, p=2, dim=-1)
#
#         N = inputs.size(0)
#         id_num = N // 2 // self.num_pos
#
#         is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())
#         is_neg_c2i = is_neg[::self.num_pos, :].chunk(2, 0)[0]  # mask [id_num, N]
#
#         centers = []
#         for i in range(id_num):
#             centers.append(inputs[targets == targets[i * self.num_pos]].mean(0))
#
#         centers = torch.stack(centers)
#
#         dist_mat = pdist_torch(centers, inputs)  #  c-i
#
#         an = dist_mat * is_neg_c2i
#         an = an[an > 1e-6].view(id_num, -1)
#         an_hard = torch.min(an,dim=1)[0]
#         #an_mean = torch.mean(torch.min(an,dim=1)[0])
#
#         ap = dist_mat * ~is_neg_c2i
#         #print(an.view(8,-1).shape)
#         ap_hard = torch.max(ap.view(8,-1),dim=1)[0]
#         #ap_mean = torch.mean(ap_hard)
#
#         triplet = ap_hard - an_hard + 0.3
#         loss = triplet[triplet>0].sum() / 8
#
#         #loss = ap_mean / an_mean
#
#         return loss
#

# 选两个最近的构成center
# class DCL(nn.Module):
#     def __init__(self, num_pos=4, feat_norm='no'):
#         super(DCL, self).__init__()
#         self.num_pos = num_pos
#         self.feat_norm = feat_norm
#
#     def forward(self,inputs, targets):
#         if self.feat_norm == 'yes':
#             inputs = F.normalize(inputs, p=2, dim=-1)
#
#         N = inputs.size(0)
#         id_num = N // 2 // self.num_pos
#
#         is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())
#         is_neg_c2i = is_neg[::self.num_pos, :].chunk(2, 0)[0]  # mask [id_num, N]
#
#         centers = []
#         for i in range(id_num):
#             # feat_i = inputs[targets == targets[i * self.num_pos]] # ([8, 768])
#             # center_8 = feat_i.mean(0) # 768
#             # dist_mat_c_i = pdist_torch(center_8.unsqueeze(0), feat_i)
#             # idx = torch.topk(dist_mat_c_i.squeeze(),1,largest=False)[1]
#             # centers.append(feat_i[idx].mean(0))
#             #-------------------------------------------------------------------
#             #index2 = torch.LongTensor(random.sample(range(0, 8), 7))  # 105 210
#             #index2 = index2.cuda().detach()
#             #centers.append(inputs[targets == targets[i * self.num_pos]][index2].mean(0)) # 随机N个构成center
#             # -------------------------------------------------------------------
#             centers.append(inputs[targets == targets[i * self.num_pos]].mean(0))  #8个构成center
#         centers = torch.stack(centers)
#
#         dist_mat = pdist_torch(centers, inputs)  #  c-i
#
#         an = dist_mat * is_neg_c2i
#         an = an[an > 1e-6].view(id_num, -1)
#
#         d_neg = torch.mean(an, dim=1, keepdim=True)
#         mask_an = (an - d_neg).expand(id_num, N - 2 * self.num_pos).lt(0)  # mask  lt
#         an = an * mask_an
#
#         list_an = []
#         for i in range (id_num):
#             list_an.append(torch.mean(an[i][an[i]>1e-6]))
#         an_mean = sum(list_an) / len(list_an)
#
#         ap = dist_mat * ~is_neg_c2i
#
#         ap_mean = torch.mean(ap[ap>1e-6])
#         loss = ap_mean / an_mean
#
#         return loss
# # 最难一个的DCL
# class DCL(nn.Module):
#     def __init__(self, num_pos=4, feat_norm='no'):
#         super(DCL, self).__init__()
#         self.num_pos = num_pos
#         self.feat_norm = feat_norm
#
#     def forward(self,inputs, targets):
#         if self.feat_norm == 'yes':
#             inputs = F.normalize(inputs, p=2, dim=-1)
#
#         N = inputs.size(0)
#         id_num = N // 2 // self.num_pos
#
#         is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())
#         is_neg_c2i = is_neg[::self.num_pos, :].chunk(2, 0)[0]  # mask [id_num, N]
#
#         centers = []
#         for i in range(id_num):
#             # index2 = torch.LongTensor(random.sample(range(0, 8), 1))  # 105 210
#             # index2 = index2.cuda().detach()
#             # centers.append(inputs[targets == targets[i * self.num_pos]][index2].mean(0))
#             centers.append(inputs[targets == targets[i * self.num_pos]].mean(0))
#         centers = torch.stack(centers)
#
#         dist_mat = pdist_torch(centers, inputs)  #  c-i
#
#         an = dist_mat * is_neg_c2i
#         an = an[an > 1e-6].view(id_num, -1)
#         # # print(an)
#         # # exit()
#         #d_neg = torch.mean(an, dim=1, keepdim=True)
#         #mask_an = (an - d_neg).expand(id_num, N - 2 * self.num_pos).lt(0)  # mask  lt
#         #an = an * mask_an
#
#
#         #an = an[:,0]
#         #print(an.shape)
#
#         # list_an = []
#         # for i in range (id_num):
#         #     list_an.append(torch.mean(an[i][an[i]>1e-6]))
#         # an_mean = sum(list_an) / len(list_an)
#
#         an_mean = torch.mean(torch.topk(an,4,largest=False)[0])
#         #an_mean = torch.mean(torch.min(an,dim=1)[0])
#
#         ap = dist_mat * ~is_neg_c2i
#         #print(an.view(8,-1).shape)
#         #ap_hard = torch.min(ap.view(8,-1),dim=1)[0]
#         ap_hard = torch.mean(torch.topk(ap.view(8,-1),4,largest=True)[0])
#         ap_mean = torch.mean(ap_hard)
#
#         #ap_mean = torch.mean(ap[ap>1e-6])
#         loss = ap_mean / an_mean.detach()
#
#         return loss
#
# class DCL(nn.Module):
#     def __init__(self, num_pos=4, feat_norm='no'):
#         super(DCL, self).__init__()
#         self.num_pos = num_pos
#         self.feat_norm = feat_norm
#
#     def forward(self,inputs, targets):
#         if self.feat_norm == 'yes':
#             inputs = F.normalize(inputs, p=2, dim=-1)
#
#         N = inputs.size(0)
#         id_num = N // 2 // self.num_pos
#
#         is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())
#         is_neg_c2i = is_neg[::self.num_pos, :].chunk(2, 0)[0]  # mask [id_num, N]
#
#         centers = []
#         for i in range(id_num):
#             centers.append(inputs[targets == targets[i * self.num_pos]].mean(0))
#         centers = torch.stack(centers)
#
#         dist_mat = pdist_torch(centers, inputs)  # c-i
#         dist_mat_an = dist_mat * is_neg_c2i
#         dist_mat_an = dist_mat_an[dist_mat_an>1e-6].view(8,-1)
#
#         list_an = []
#
#         for i in range(8):
#             idx_an = torch.nonzero(is_neg_c2i[i] > 0.5)
#             feat_an = inputs[idx_an]  # 56, 768
#             _, idx = torch.topk(dist_mat_an[i],1,largest=False)
#             feat_proxy = torch.mean(feat_an[idx],dim=0)
#             dist_an = pdist_torch(centers[i].view(1,-1), feat_proxy.view(1,-1))
#             list_an.append(dist_an)
#
#         an_mean = sum(list_an) / len(list_an)
#
#         ap = dist_mat * ~is_neg_c2i
#         ap_mean = torch.mean(ap[ap>1e-6])
#         loss = ap_mean / an_mean
#
#         return loss

# # c-c (AN)
# class DCL(nn.Module):
#     def __init__(self, num_pos=4, feat_norm='no'):
#         super(DCL, self).__init__()
#         self.num_pos = num_pos
#         self.feat_norm = feat_norm
#
#     def forward(self,inputs, targets):
#         if self.feat_norm == 'yes':
#             inputs = F.normalize(inputs, p=2, dim=-1)
#
#         N = inputs.size(0)
#         id_num = N // 2 // self.num_pos
#
#         is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())
#         is_neg_c2i = is_neg[::self.num_pos, :].chunk(2, 0)[0]  # mask [id_num, N]
#
#         targets_32 = targets[:32]
#         targets_half = targets_32[::4]
#         is_neg_c2c = targets_half.expand(8, 8).ne(targets_half.expand(8, 8).t())
#
#         centers = []
#         for i in range(id_num):
#             centers.append(inputs[targets == targets[i * self.num_pos]].mean(0))
#         centers = torch.stack(centers)
#
#         dist_mat = pdist_torch(centers, inputs)  # c-i
#         dist_mat_cc = pdist_torch(centers, centers)  #  c-i
#
#         an = dist_mat_cc*is_neg_c2c
#         an = an[an > 1e-6].view(id_num, -1) #[8, 7]
#         an = torch.min(an,dim=1)[0]
#
#         # d_neg = torch.mean(an, dim=1, keepdim=True)
#         # mask_an = (an - d_neg).expand(id_num, 7).lt(0)  # mask  lt
#         # an = an * mask_an
#
#
#         list_an = []
#         for i in range (id_num):
#             list_an.append(torch.mean(an[i][an[i]>1e-6]))
#         an_mean = sum(list_an) / len(list_an)
#
#         ap = dist_mat * ~is_neg_c2i
#         ap_mean = torch.mean(ap[ap>1e-6])
#
#         loss = ap_mean / an_mean
#
#         return loss


# # c4-c4 (AN)
# class DCL(nn.Module):
#     def __init__(self, num_pos=4, feat_norm='no'):
#         super(DCL, self).__init__()
#         self.num_pos = num_pos
#         self.feat_norm = feat_norm
#
#     def forward(self,inputs, targets):
#         if self.feat_norm == 'yes':
#             inputs = F.normalize(inputs, p=2, dim=-1)
#
#         N = inputs.size(0)
#         id_num = N // 2 // self.num_pos
#
#         targets_32 = targets[:32]
#         targets_half = targets[::4] # 16
#         is_neg_c2c = targets_half.expand(16, 16).ne(targets_half.expand(16, 16).t())
#
#         feat_rgb,feat_ir = inputs.chunk(2,0)
#
#         centers_rgb = []
#         centers_ir = []
#         for i in range(id_num):
#             centers_rgb.append(feat_rgb[targets_32 == targets_32[i * self.num_pos]].mean(0))
#             centers_ir.append(feat_ir[targets_32 == targets_32[i * self.num_pos]].mean(0))
#         centers_rgb = torch.stack(centers_rgb) # [8, 768]
#         centers_ir = torch.stack(centers_ir)
#
#         ap = torch.diag(pdist_torch(centers_rgb, centers_ir))
#         ap_mean = torch.mean(ap)
#         dist_mat_cc = pdist_torch(torch.cat([centers_rgb,centers_ir]),torch.cat([centers_rgb,centers_ir]))  # [16, 16] c-i
#         an = dist_mat_cc*is_neg_c2c
#         #print(an[an > 1e-6].shape)
#         an = torch.min(an[an > 1e-6].view(id_num*2, -1),dim=1)[0]
#
#         #an = torch.topk(an,10,largest=False)[0]
#         an_mean = torch.mean(an)
#
#         loss = ap_mean / an_mean
#         #loss = torch.mean(ap/an)*0.5 + ap_mean / an_mean*0.5
#         return loss


# class DCL(nn.Module):
#     def __init__(self, num_pos=4, feat_norm='no'):
#         super(DCL, self).__init__()
#         self.num_pos = num_pos
#         self.feat_norm = feat_norm
#
#     def forward(self,inputs, targets):
#         if self.feat_norm == 'yes':
#             inputs = F.normalize(inputs, p=2, dim=-1)
#
#         N = inputs.size(0)
#         id_num = N // 2 // self.num_pos
#
#         targets_32 = targets[:32]
#         targets_half = targets[::4] # 16
#         is_neg_c2c = targets_half.expand(16, 16).ne(targets_half.expand(16, 16).t())
#
#         feat_rgb,feat_ir = inputs.chunk(2,0)
#
#         centers_rgb = []
#         centers_ir = []
#         for i in range(id_num):
#             centers_rgb.append(feat_rgb[targets_32 == targets_32[i * self.num_pos]].mean(0))
#             centers_ir.append(feat_ir[targets_32 == targets_32[i * self.num_pos]].mean(0))
#         centers_rgb = torch.stack(centers_rgb) # [8, 768]
#         centers_ir = torch.stack(centers_ir)
#
#         ap_mean = torch.mean(torch.diag(pdist_torch(centers_rgb, centers_ir))) # ap
#
#         dist_mat_cc = pdist_torch(torch.cat([centers_rgb,centers_ir]),torch.cat([centers_rgb,centers_ir]))  # [16, 16] c-i
#         an = dist_mat_cc*is_neg_c2c
#
#         an = torch.min(an[an > 1e-6].view(id_num, -1),dim=1)[0]
#         #an = an[an > 1e-6]
#         an_mean = torch.mean(an)
#
#         loss = ap_mean / an_mean
#
#         return loss

#old c2i
# class DCL(nn.Module):
#     def __init__(self, num_pos=4, feat_norm='no'):
#         super(DCL, self).__init__()
#         self.num_pos = num_pos
#         self.feat_norm = feat_norm
#
#     def forward(self,inputs, targets):
#         if self.feat_norm == 'yes':
#             inputs = F.normalize(inputs, p=2, dim=-1)
#
#         N = inputs.size(0)
#         id_num = N // 2 // self.num_pos
#
#         is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())
#         is_neg_c2i = is_neg[::self.num_pos, :].chunk(2, 0)[0]  # mask [id_num, N]
#
#         centers = []
#         for i in range(id_num):
#             centers.append(inputs[targets == targets[i * self.num_pos]].mean(0))
#         centers = torch.stack(centers)
#
#         dist_mat = pdist_torch(centers, inputs)  #  c-i
#
#         an = dist_mat * is_neg_c2i
#         an = an[an > 1e-6].view(id_num, -1)
#
#         d_neg = torch.mean(an, dim=1, keepdim=True)
#         mask_an = (an - d_neg).expand(id_num, N - 2 * self.num_pos).lt(0)  # mask
#         an = an * mask_an
#
#         list_an = []
#         for i in range (id_num):
#             list_an.append(torch.mean(an[i][an[i]>1e-6]))
#         #
#         #an_mean = torch.mean(torch.min(an,dim=1)[0])
#
#         an_mean = sum(list_an) / len(list_an)
#         #an_mean = torch.mean(an)
#         ap = dist_mat * ~is_neg_c2i
#         ap_mean = torch.mean(ap[ap>1e-6])
#
#         loss = ap_mean / an_mean
#         return loss

# Old c-i 但只选择c-c的最难an 8个一共
# class DCL(nn.Module):
#     def __init__(self, num_pos=4, feat_norm='no'):
#         super(DCL, self).__init__()
#         self.num_pos = num_pos
#         self.feat_norm = feat_norm
#
#     def forward(self,inputs, targets):
#         if self.feat_norm == 'yes':
#             inputs = F.normalize(inputs, p=2, dim=-1)
#
#         N = inputs.size(0)
#         id_num = N // 2 // self.num_pos
#
#         targets_32 = targets[:32]
#         targets_half = targets_32[::4] # 8
#         is_neg_c2c = targets_half.expand(8, 8).ne(targets_half.expand(8, 8).t())
#
#         is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())
#         is_neg_c2i = is_neg[::self.num_pos, :].chunk(2, 0)[0]  # mask [id_num, N]
#
#         feat_rgb, feat_ir = inputs.chunk(2, 0)
#         #
#         centers_rgb = []
#         centers_ir = []
#
#         centers = []
#         for i in range(id_num):
#             centers_rgb.append(feat_rgb[targets_32 == targets_32[i * self.num_pos]].mean(0))
#             centers_ir.append(feat_ir[targets_32 == targets_32[i * self.num_pos]].mean(0))
#
#             centers.append(inputs[targets == targets[i * self.num_pos]].mean(0))
#
#         centers_rgb = torch.stack(centers_rgb)  # [8, 768]
#         centers_ir = torch.stack(centers_ir)
#         centers = torch.stack(centers)
#         ap_mean = torch.mean(torch.diag(pdist_torch(centers_rgb, centers_ir))) # ap
#
#         dist_mat = pdist_torch(centers, inputs)  #  c-i
#
#
#         dist_mat_c2c = pdist_torch(centers, centers)  # c-c  8 8
#         an = dist_mat_c2c * is_neg_c2c
#         an = an[an > 1e-6].view(id_num, -1) # 8 x 7
#         an_mean = torch.mean(torch.min(an,dim=1)[0])
#
#         #ap = dist_mat * ~is_neg_c2i
#         #ap_mean = torch.mean(ap[ap>1e-6])
#
#         loss = ap_mean / an_mean
#         return loss

#
# HCT TRI
# class DCL(nn.Module):
#     def __init__(self, num_pos=4, feat_norm='no'):
#         super(DCL, self).__init__()
#         self.num_pos = num_pos
#         self.feat_norm = feat_norm
#         self.ranking_loss = nn.SoftMarginLoss()
#
#     def forward(self,inputs, targets):
#         if self.feat_norm == 'yes':
#             inputs = F.normalize(inputs, p=2, dim=-1)
#
#         N = inputs.size(0)
#         id_num = N // 2 // self.num_pos
#
#         targets_32 = targets[:32]
#         targets_half = targets[::4] # 16
#         is_neg_c2c = targets_half.expand(16, 16).ne(targets_half.expand(16, 16).t())
#
#         feat_rgb,feat_ir = inputs.chunk(2,0)
#
#         centers_rgb = []
#         centers_ir = []
#         for i in range(id_num):
#             centers_rgb.append(feat_rgb[targets_32 == targets_32[i * self.num_pos]].mean(0))
#             centers_ir.append(feat_ir[targets_32 == targets_32[i * self.num_pos]].mean(0))
#         centers_rgb = torch.stack(centers_rgb) # [8, 768]
#         centers_ir = torch.stack(centers_ir)
#
#         ap = torch.diag(pdist_torch(centers_rgb, centers_ir))
#
#         dist_mat_cc = pdist_torch(torch.cat([centers_rgb,centers_ir]),torch.cat([centers_rgb,centers_ir]))  # [16, 16] c-i
#         an = dist_mat_cc*is_neg_c2c
#
#         an = torch.min(an[an > 1e-6].view(id_num, -1),dim=1)[0]
#
#         y = an.new().resize_as_(an).fill_(1)
#
#         loss = self.ranking_loss(an-ap, y)
#
#         return loss