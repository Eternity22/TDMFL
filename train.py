import sys
from dataloader import SYSUData, RegDBData,TestData, GenIdx, IdentitySampler
from datamanager import process_gallery_sysu, process_query_sysu, process_test_regdb
import numpy as np
import torch.utils.data as data
from torch.autograd import Variable
import torch
from torch.cuda import amp
import torch.nn as nn
import os.path as osp
import os
from loss.HC import hetero_loss
from loss.HCT import CenterTripletLoss, CenterTripletLoss1
from loss.Triplet import TripletLoss_init
from loss.Triplet_z import  TripletLoss_Balanced_2branch
from model.make_model3 import build_vision_transformer
import time
import optimizer
from model.vision_transformer import weights_init_classifier
from scheduler import create_scheduler
from utils import AverageMeter, set_seed
from transforms import transform_rgb, transform_rgb2gray, transform_thermal, transform_test
from optimizer import make_optimizer
from config.config import cfg
from eval_metrics import eval_sysu, eval_regdb
import argparse
torch.cuda.empty_cache()
parser = argparse.ArgumentParser(description="TDMFL Training")
parser.add_argument('--config_file', default='config/SYSU-6block.yml',
                    help='path to config file', type=str)
parser.add_argument('--trial', default=1,
                    help='only for RegDB', type=int)
parser.add_argument('--resume', '-r', default='',
                    help='resume from checkpoint', type=str)
parser.add_argument('--model_单项path', default='save_model/',
                    help='model save path', type=str)
parser.add_argument('--num_workers', default=0,
                    help='number of data loading workers', type=int)
parser.add_argument('--start_test', default=0,
                    help='start to test in training', type=int)
parser.add_argument('--test_batch', default=128,
                    help='batch size for test', type=int)
parser.add_argument('--test_epoch', default=2,
                    help='test model every 2 epochs', type=int)
parser.add_argument('--save_epoch', default=2,
                    help='save model every 2 epochs', type=int)
parser.add_argument('--gpu', default='0',
                    help='gpu device ids for CUDA_VISIBLE_DEVICES', type=str)
parser.add_argument("opts", help="Modify config options using the command-line",
                    default=None,nargs=argparse.REMAINDER)
args = parser.parse_args()
t = time.strftime("-%Y%m%d-%H%M%S", time.localtime())  # 时间戳
filename = './logs/'+ os.path.splitext(args.config_file[7:])[0]+'-log' + t + '.txt'
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")  # 防止编码错误

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
log = Logger(filename)
sys.stdout = log
print(args.config_file)
if args.config_file != '':
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

set_seed(cfg.SEED)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()


if cfg.DATASET == 'sysu':
    data_path = cfg.DATA_PATH_SYSU
    trainset_gray = SYSUData(data_path, transform1=transform_rgb2gray, transform2=transform_thermal)
    color_pos_gray, thermal_pos_gray = GenIdx(trainset_gray.train_color_label, trainset_gray.train_thermal_label)

    trainset_rgb = SYSUData(data_path, transform1=transform_rgb, transform2=transform_thermal)
    color_pos_rgb, thermal_pos_rgb = GenIdx(trainset_rgb.train_color_label, trainset_rgb.train_thermal_label)

elif cfg.DATASET == 'regdb':
    data_path = cfg.DATA_PATH_RegDB
    trainset_gray = RegDBData(data_path, args.trial, transform1=transform_rgb2gray,transform2=transform_thermal)
    color_pos_gray, thermal_pos_gray = GenIdx(trainset_gray.train_color_label, trainset_gray.train_thermal_label)

    trainset_rgb = RegDBData(data_path, args.trial, transform1=transform_rgb, transform2=transform_thermal)
    color_pos_rgb, thermal_pos_rgb = GenIdx(trainset_rgb.train_color_label, trainset_rgb.train_thermal_label)
    print('Current trial :', args.trial)


num_classes = len(np.unique(trainset_rgb.train_color_label))
model = build_vision_transformer(num_classes=num_classes,cfg = cfg)
net_modal_classifier1 = nn.Linear(768, 3, bias=False)
net_modal_classifier1.apply(weights_init_classifier)
net_modal_classifier1.to(device)
model.to(device)

# load checkpoint
if len(args.resume) > 0:
    model_path = args.model_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        model.load_param(model_path)
        path = args.model_path + args.resume
        path = path[:-4]+'_modal_classifier.pth'
        net_modal_classifier1.load_state_dict(torch.load(path), strict=False)
        print('==> loaded checkpoint {}'.format(path))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# Loss
criterion_ID = nn.CrossEntropyLoss()
criterion_Tri_Balanced = TripletLoss_Balanced_2branch(margin=cfg.MARGIN, feat_norm='no')
criterion_Tri_init = TripletLoss_init(margin=cfg.MARGIN, feat_norm='no')
criterion_HC = hetero_loss(margin=0)
criterion_HCT = CenterTripletLoss()

optimizer = make_optimizer(cfg, model)
scheduler = create_scheduler(cfg, optimizer)
modal_classifier_optimizer_1 = torch.optim.AdamW(net_modal_classifier1.parameters(), lr=cfg.BASE_LR, weight_decay=cfg.WEIGHT_DECAY)
modal_classifier_scheduler_1 = create_scheduler(cfg, modal_classifier_optimizer_1)

scaler = amp.GradScaler()


if cfg.DATASET == 'sysu':   # for test
    query_img, query_label, query_cam = process_query_sysu(data_path, mode='all')
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))

    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode='all', trial=0, gall_mode='single')
    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))

elif cfg.DATASET == 'regdb':
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))

    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')
    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))

# Test loader
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers)
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers)


loss_meter = AverageMeter()
loss_ce_meter = AverageMeter()
loss_tri_meter = AverageMeter()
acc_rgb_meter = AverageMeter()
acc_ir_meter = AverageMeter()
tri_num_g_meter = AverageMeter()
tri_num_l1_meter = AverageMeter()
tri_num_l2_meter = AverageMeter()
loader_batch = 32
print('Current Epoch parament：{}'.format(cfg.Epo))
print('Current Block Num parament：{}'.format(cfg.BLOCK_NUM))

def train(epoch):
    start_time = time.time()

    loss_meter.reset()
    loss_ce_meter.reset()
    loss_tri_meter.reset()
    acc_rgb_meter.reset()
    acc_ir_meter.reset()
    tri_num_g_meter.reset()
    tri_num_l1_meter.reset()
    tri_num_l2_meter.reset()
    scheduler.step(epoch)
    model.train()

    for idx, (input1, input2, label1, label2) in enumerate(trainloader):

        optimizer.zero_grad()
        input1 = input1.to(device)
        input2 = input2.to(device)
        label1 = label1.to(device)
        label2 = label2.to(device)
        labels = torch.cat((label1,label2),0)
        modal_v_labels = Variable(torch.ones(loader_batch).long().cuda())
        modal_t_labels = Variable(torch.zeros(loader_batch).long().cuda())
        modal_3_labels = Variable(2 * torch.ones(loader_batch).long().cuda())
        with amp.autocast(enabled=True):
            cls_score_b1_main, cls_score_b1_main_1, cls_score_b1_main_2, \
            features_b1_main, features_b1_main_1, features_b1_main_2, \
            cls_score_b2_adv, \
            features_b2_adv, \
            feat_b2_adv = model(torch.cat([input1,input2]))

            score1_main, score2_main = cls_score_b1_main.chunk(2,0)
            feat1_main, feat2_main = features_b1_main.chunk(2,0)
            loss_id_main = criterion_ID(cls_score_b1_main, labels.long())+ criterion_ID(cls_score_b1_main_1, labels.long())  + criterion_ID(cls_score_b1_main_2, labels.long())
            score1_adv, score2_adv = cls_score_b2_adv.chunk(2,0)
            feat1_main, feat2_main = features_b2_adv.chunk(2,0)


            out_modal_adv = net_modal_classifier1(feat_b2_adv.detach())
            if epoch <= cfg.Epo:
                loss_id = loss_id_main #+ loss_id_adv
                features_all = torch.cat((features_b1_main,features_b2_adv),0)
                labels_all = torch.cat((labels,labels),0)
                loss_tri = criterion_Tri_init(features_b1_main, features_b1_main, labels) + criterion_Tri_init(features_b1_main_1, features_b1_main_1, labels) + criterion_Tri_init(features_b1_main_2, features_b1_main_2, labels)
                loss_ht = criterion_HCT(features_b2_adv, labels)
                loss_tri_global, num_g = criterion_Tri_init(features_b1_main, features_b1_main, labels)
                loss_tri_local1, num_l1 = criterion_Tri_init(features_b1_main_1, features_b1_main_1, labels)
                loss_tri_local2, num_l2 = criterion_Tri_init(features_b1_main_2, features_b1_main_2, labels)
                loss_tri = loss_tri_global / 3.0 + loss_tri_local1 / 3.0 + loss_tri_local2 / 3.0

                loss_total = loss_id/3.0 + loss_ht + loss_tri


                tri_num_l1_meter.update(num_l1)
                tri_num_l2_meter.update(num_l2)
                tri_num_g_meter.update(num_g)
                optimizer.zero_grad()
                scaler.scale(loss_total).backward()
                scaler.step(optimizer)

                modal_loss = criterion_ID(out_modal_adv[:loader_batch], modal_v_labels) + criterion_ID(out_modal_adv[loader_batch:], modal_t_labels)
                modal_classifier_optimizer_1.zero_grad()
                modal_loss.backward()
                modal_classifier_optimizer_1.step()
                if (idx + 1) % 32 == 0:
                    print('modal_loss: ' + str(modal_loss.cpu().detach().numpy()))
            else:
                modal_loss = criterion_ID(out_modal_adv[:loader_batch], modal_v_labels) + criterion_ID(out_modal_adv[loader_batch:], modal_t_labels)

                modal_classifier_optimizer_1.zero_grad()
                modal_loss.backward()
                modal_classifier_optimizer_1.step()

                out2 = net_modal_classifier1(feat_b2_adv)
                loss2 = criterion_ID(out2[:loader_batch], modal_3_labels) + criterion_ID(out2[loader_batch:],
                                                                                         modal_3_labels)
                loss_id = loss_id_main
                features_all = torch.cat((features_b1_main,features_b2_adv), 0)

                labels_all = torch.cat((labels, labels), 0)
                # loss_tri = criterion_Tri(features_b1_main, features_b1_main, labels) + criterion_Tri(features_b1_main, features_b2_adv, labels) #64*128  / b1*b1  b1*b2
                # loss_tri = criterion_Tri(features_b1_main, features_b2_adv, labels)
                loss_tri, num_g = criterion_Tri_Balanced(features_all, features_all, labels_all)
                loss_total = loss_id/3.0 + loss2 + loss_tri
                tri_num_g_meter.update(num_g)

                optimizer.zero_grad()
                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                if (idx + 1) % 32 == 0:
                    print('modal_loss: ' + str(modal_loss.cpu().detach().numpy()))
                    print('loss2: ' + str(loss2.cpu().detach().numpy()))
            scaler.update()


        acc_rgb = (score1_main.max(1)[1] == label1).float().mean()
        acc_ir = (score2_main.max(1)[1] == label2).float().mean()

        loss_tri_meter.update(loss_tri.item())
        loss_ce_meter.update(loss_id.item())
        loss_meter.update(loss_total.item())

        acc_rgb_meter.update(acc_rgb, 1)
        acc_ir_meter.update(acc_ir, 1)

        torch.cuda.synchronize()

        if (idx + 1) % 32 == 0 :

            print('Epoch[{}] Iteration[{}/{}]'
                  ' Loss: {:.3f}, Tri:{:.3f} CE:{:.3f}, '
                  'Acc_RGB: {:.3f}, Acc_IR: {:.3f}, '
                  'Base Lr: {:.2e} '.format(epoch, (idx+1),
                len(trainloader), loss_meter.avg, loss_tri_meter.avg,
                loss_ce_meter.avg, acc_rgb_meter.avg, acc_ir_meter.avg,
                optimizer.state_dict()['param_groups'][0]['lr']))

    end_time = time.time()
    time_per_batch = end_time - start_time
    print(' Epoch {} done. Time per batch: {:.1f}[min] '.format(epoch, time_per_batch/60))


def test(query_loader, gall_loader, dataset = 'sysu'):
    model.eval()
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            input = Variable(input.cuda())
            feat1 = model(input)
            dim1 = feat1.shape[1]

            break

    nquery = len(query_label)
    ngall = len(gall_label)
    print('Testing...')
    ptr = 0
    gall_feat = np.zeros((ngall, dim1))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = model(input)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num

    ptr = 0
    query_feat = np.zeros((nquery, dim1))

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = model(input)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()

            ptr = ptr + batch_num

    distmat = -np.matmul(query_feat, np.transpose(gall_feat))

    if dataset == 'sysu':
        cmc, mAP, mInp = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)

    else:
        cmc, mAP, mInp = eval_regdb(distmat, query_label, gall_label)


    return cmc, mAP, mInp


# Training
best_mAP = 0
print('==> Start Training...')
for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH + 1):

    print('==> Preparing Data Loader...')

    sampler_rgb = IdentitySampler(trainset_rgb.train_color_label, trainset_rgb.train_thermal_label,
                                  color_pos_rgb,thermal_pos_rgb, cfg.BATCH_SIZE, per_img=cfg.NUM_POS)

    # RGB-IR
    trainset_rgb.cIndex = sampler_rgb.index1  # color index
    trainset_rgb.tIndex = sampler_rgb.index2

    trainloader = data.DataLoader(trainset_rgb, batch_size=cfg.BATCH_SIZE, sampler=sampler_rgb,
                                   num_workers=args.num_workers, drop_last=True, pin_memory=True)

    train(epoch)

    if epoch > args.start_test and epoch % args.test_epoch == 0:
    # if epoch > -1:
        cmc, mAP, mInp = test(query_loader, gall_loader, cfg.DATASET



                              )
        print('mAP: {:.2%} | mInp:{:.2%} | top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(mAP,mInp,cmc[0],cmc[4],cmc[9],cmc[19]))

        if mAP > best_mAP:
            best_mAP = mAP
            if cfg.DATASET == 'sysu':
                torch.save(modal_classifier_optimizer_1.state_dict(),  osp.join('./save_model', os.path.basename(args.config_file)[:-4] + '_best_modal_classifier.pth'))
                torch.save(model.state_dict(), osp.join('./save_model', os.path.basename(args.config_file)[:-4] + '_best.pth'))  # maybe not the best
            else:
                torch.save(model.state_dict(), osp.join('./save_model', os.path.basename(args.config_file)[:-4] + '_best_trial_{}.pth'.format(args.trial)))

    if epoch % 2 == 0:
        torch.save(modal_classifier_optimizer_1.state_dict(),
                   osp.join('./save_model', os.path.basename(args.config_file)[:-4] + '_epoch{}_modal_classifier.pth'.format(epoch)))
        torch.save(model.state_dict(), osp.join('./save_model', os.path.basename(args.config_file)[:-4]  + '_epoch{}.pth'.format(epoch)))






