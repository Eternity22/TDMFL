import copy

from model.vision_transformer import ViT
import torch
import torch.nn as nn

# L2 norm
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class build_vision_transformer(nn.Module):
    def __init__(self, num_classes, cfg):
        super(build_vision_transformer, self).__init__()
        self.in_planes = 768

        self.base = ViT(img_size=[cfg.H,cfg.W],
                        stride_size=cfg.STRIDE_SIZE,
                        drop_path_rate=cfg.DROP_PATH,
                        drop_rate=cfg.DROP_OUT,
                        attn_drop_rate=cfg.ATT_DROP_RATE,
                       cfg = cfg)

        self.base.load_param(cfg.PRETRAIN_PATH)
        print('Loading pretrained ImageNet model......from {}'.format(cfg.PRETRAIN_PATH))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.classifier1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier1.apply(weights_init_classifier)

        self.classifier2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier2.apply(weights_init_classifier)

        self.classifier21 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier21.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck1.bias.requires_grad_(False)
        self.bottleneck1.apply(weights_init_kaiming)

        self.bottleneck2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck2.bias.requires_grad_(False)
        self.bottleneck2.apply(weights_init_kaiming)

        self.bottleneck21 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck21.bias.requires_grad_(False)
        self.bottleneck21.apply(weights_init_kaiming)


        layer_norm = self.base.norm
        self.layer_norm1 = copy.deepcopy(layer_norm)
        self.layer_norm2 = copy.deepcopy(layer_norm)
        self.layer_norm3 = copy.deepcopy(layer_norm)

        self.layer_norm21 = copy.deepcopy(layer_norm)
        blocks_to_copy = self.base.blocks[cfg.BLOCK_NUM: 12]

        self.b1 = nn.Sequential(*[copy.deepcopy(block) for block in blocks_to_copy])

        self.b2 = nn.Sequential(*[copy.deepcopy(block) for block in blocks_to_copy])

        self.l2norm = Normalize(2)


    def forward(self, x):
        x10, x10_1, x10_2, x9, x9_1, x9_2 = self.base(x)

        if self.training:

            features_b1_main = self.layer_norm1(self.b1(x10)[:,0])
            features_b1_main_1 = self.layer_norm1(self.b1(x10_1)[:,0])
            features_b1_main_2 = self.layer_norm1(self.b1(x10_2)[:,0])

            features_b2_adv = self.layer_norm21(self.b2(x10)[:,0])

            feat_b1_main = self.bottleneck(features_b1_main)
            feat_b1_main_1 = self.bottleneck1(features_b1_main_1)
            feat_b1_main_2 = self.bottleneck2(features_b1_main_2)

            feat_b2_adv = self.bottleneck21(features_b2_adv)

            cls_score_b1_main = self.classifier(feat_b1_main)
            cls_score_b1_main_1 = self.classifier1(feat_b1_main_1)
            cls_score_b1_main_2 = self.classifier2(feat_b1_main_2)

            cls_score_b2_adv = self.classifier21(feat_b2_adv)


            return cls_score_b1_main, cls_score_b1_main_1, cls_score_b1_main_2, \
                   features_b1_main, features_b1_main_1, features_b1_main_2, \
                   cls_score_b2_adv,\
                   features_b2_adv,  \
                   feat_b2_adv

        else:

            features_b1_main = self.layer_norm1(self.b1(x10)[:, 0])
            features_b1_main_1 = self.layer_norm1(self.b1(x10_1)[:,0])
            features_b1_main_2 = self.layer_norm1(self.b1(x10_2)[:,0])
            features_b2_adv = self.layer_norm21(self.b2(x10)[:, 0])

            feat_b1 = self.bottleneck(features_b1_main)
            feat_b11 = self.bottleneck1(features_b1_main_1)
            feat_b12 = self.bottleneck2(features_b1_main_2)
            feat_b2 = self.bottleneck21(features_b2_adv)


            feat_all = torch.cat([feat_b1, feat_b11/2,feat_b12/2,feat_b2], dim=1)
            return self.l2norm(feat_all)


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:


            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:

            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))