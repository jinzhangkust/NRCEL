"""
Author: Dr. Jin Zhang
E-mail: j.zhang@kust.edu.cn
Dept: Kunming University of Science and Technology
Created on 2025.03.22
"""
import random

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.resnet_proj import resnet50
from models.inception_proj import inception
from models.ViT import vit_b_16_diy    # torchvision导入模型需将图片下采样到224*224
from sklearn.ensemble import IsolationForest

import os
import sys
import time
import argparse
import numpy as np

from data import get_froth_data, data4cls

from util import AverageMeter, accuracy


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=80, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers=4*num_GPU')
    parser.add_argument('--epoch', type=int, default=100, help='number of training epochs')
    parser.add_argument('--warmup_epoch', type=int, default=100, help='number of training epochs')
    parser.add_argument('--total_epochs', type=int, default=600, help='number of training epochs')
    parser.add_argument('--load_epoch', type=int, default=300, help='number of training epochs')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    # model dataset
    parser.add_argument('--model_name', type=str, default='iForest_50')
    # checkpoint
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    # create
    opt = parser.parse_args()
    # save
    opt.save_folder = os.path.join('./save', opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#os.environ["CUDA_LAUNCH_BLOCKING"] = '1'


vggmodel = torchvision.models.vgg16(weights=None)
class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        self.feature = vggmodel.features
        self.feature.add_module('pool', nn.AdaptiveAvgPool2d((1, 1)))
        self.projector = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True))
        self.classifier = nn.Linear(128, 6)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        #print(f"x: {x.size()}")
        code = self.projector(x)
        out = self.classifier(code)
        return code, out


class VGGSensor(nn.Module):
    def __init__(self):
        super(VGGSensor, self).__init__()

        self.feature = vggmodel.features
        self.feature.add_module('pool', nn.AdaptiveAvgPool2d((1, 1)))
        self.predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)

    def forward(self, x):
        x = self.feature(x)
        x = self.predictor(x.view(x.size(0), -1))
        return x

class InceptionSensor(nn.Module):
    def __init__(self):
        super(InceptionSensor, self).__init__()

        self.feature = inception()
        self.feature.add_module('pool', nn.AdaptiveAvgPool2d((1, 1)))
        self.predictor = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)

    def forward(self, x):
        x = self.feature(x)
        x = self.predictor(x.view(x.size(0), -1))
        return x

class ResNetSensor(nn.Module):
    def __init__(self):
        super(ResNetSensor, self).__init__()

        self.feature = resnet50()
        self.feature.add_module('pool', nn.AdaptiveAvgPool2d((1, 1)))
        self.predictor = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)

    def forward(self, x):
        x = self.feature(x)
        x = self.predictor(x.view(x.size(0), -1))
        return x


def set_loader(opt):
    full_data = data4cls()
    train_size = int(0.6 * len(full_data))
    val_size = int(0.2 * len(full_data))
    test_size = len(full_data) - train_size - val_size
    train_data, val_data, test_data = torch.utils.data.random_split(full_data, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    random.seed(42)
    num_noise = int(0.4 * train_size)
    noise_idx = random.sample(range(train_size), num_noise)
    for idx in noise_idx:
        train_data.dataset.labels[idx] = random.randint(0, 5)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    val_loader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    test_loader = DataLoader(test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    return train_data, train_loader, val_loader, test_loader


def clean_noise_division(model, train_data, opt, n_classes=6):
    #dataset = train_loader.dataset  # indics_set存储的是fulldata中的绝对路径，超出了traindata的索引范围
    full_data = data4cls()
    """train_size = int(0.6 * len(full_data))
    val_size = int(0.2 * len(full_data))
    test_size = len(full_data) - train_size - val_size
    train_data, val_data, test_data = torch.utils.data.random_split(full_data, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    random.seed(42)
    num_noise = int(0.4 * train_size)
    noise_idx = random.sample(range(train_size), num_noise)
    for idx in noise_idx:
        train_data.dataset.labels[idx] = random.randint(0, 5)"""
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    feature_set = None
    label_set = None
    indics_set = None
    noise_data = None
    clean_data = None
    #noise_mask = np.zeros(len(train_loader), dtype=bool)
    for _, (idx, (im_w, _), labels) in enumerate(train_loader):
        im_w = im_w.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        # compute output
        feature, _ = model(im_w)
        if feature_set is not None:
            feature_set = np.append(feature_set, feature.cpu().detach().numpy(), axis=0)
            label_set = np.append(label_set, labels.cpu().detach().numpy(), axis=0)
            indics_set = np.append(indics_set, idx.cpu().detach().numpy(), axis=0)  # global indics in fulldata
        else:
            feature_set = feature.cpu().detach().numpy()
            label_set = labels.cpu().detach().numpy()
            indics_set = idx.cpu().detach().numpy()
    print(f"Num of training data: {len(indics_set)}")
    for cls in range(n_classes):
        cls_mask = label_set == cls
        cls_feature = feature_set[cls_mask]
        #cls_indics = np.where(cls_mask)[0]
        cls_indics = indics_set[cls_mask]
        print(f"cls_indics: {cls_indics}")
        #ins_mask = np.where(cls_mask)[0]

        iforest = IsolationForest(contamination=0.4)    # 40% noise
        iforest.fit(cls_feature)
        anomalies = iforest.predict(cls_feature)
        print(f"anomalies: {anomalies}")
        noise_indics = cls_indics[anomalies == -1]
        clean_indics = cls_indics[anomalies == 1]
        if noise_data is not None:
            noise_data = np.append(noise_data, noise_indics)
            clean_data = np.append(clean_data, clean_indics)
        else:
            noise_data = noise_indics
            clean_data = clean_indics
        print(f"Class {cls} has {len(noise_indics)} noise samples.")
    # divide clean data and noise data
    #clean_mask = ~noise_mask
    #clean_data = indics_set[clean_mask]
    #noise_data = indics_set[noise_mask]

    clean_data_sampler = torch.utils.data.SubsetRandomSampler(clean_data)
    noise_data_sampler = torch.utils.data.SubsetRandomSampler(noise_data)
    clean_loader = DataLoader(full_data, batch_size=opt.batch_size, sampler=clean_data_sampler)
    noise_loader = DataLoader(full_data, batch_size=opt.batch_size, sampler=noise_data_sampler)
    return clean_data, clean_loader, noise_loader


def set_trimodel():
    vggsensor = VGGSensor()
    inceptionsensor = InceptionSensor()
    resnetsensor = ResNetSensor()
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        vggsensor = vggsensor.cuda()
        inceptionsensor = inceptionsensor.cuda()
        resnetsensor = resnetsensor.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    return vggsensor, inceptionsensor, resnetsensor, criterion

def set_trioptimizer(opt, vggsensor, inceptionsensor, resnetsensor):
    optimizer_vgg = optim.Adam(vggsensor.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    optimizer_inception = optim.Adam(inceptionsensor.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    optimizer_resnet = optim.Adam(resnetsensor.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    return optimizer_vgg, optimizer_inception, optimizer_resnet


def pseudo_loader(opt, clean_data, aug_dict_vgg, aug_dict_inception, aug_dict_resnet):
    full_data = data4cls()
    train_size = int(0.6 * len(full_data))
    val_size = int(0.2 * len(full_data))
    test_size = len(full_data) - train_size - val_size
    train_data, val_data, test_data = torch.utils.data.random_split(full_data, [train_size, val_size, test_size],
                                                                    generator=torch.Generator().manual_seed(42))

    train_labeled_pseudo_index_vgg = clean_data.tolist() + list(aug_dict_vgg['pseudo_idx'])
    train_labeled_pseudo_vgg_sampler = torch.utils.data.SubsetRandomSampler(train_labeled_pseudo_index_vgg)
    train_labeled_pseudo_vgg_loader = DataLoader(full_data, batch_size=opt.batch_size, sampler=train_labeled_pseudo_vgg_sampler)

    train_labeled_pseudo_index_inception = clean_data.tolist() + list(aug_dict_inception['pseudo_idx'])
    train_labeled_pseudo_inception_sampler = torch.utils.data.SubsetRandomSampler(train_labeled_pseudo_index_inception)
    train_labeled_pseudo_inception_loader = DataLoader(full_data, batch_size=opt.batch_size, sampler=train_labeled_pseudo_inception_sampler)

    train_labeled_pseudo_index_resnet = clean_data.tolist() + list(aug_dict_resnet['pseudo_idx'])
    train_labeled_pseudo_resnet_sampler = torch.utils.data.SubsetRandomSampler(train_labeled_pseudo_index_resnet)
    train_labeled_pseudo_resnet_loader = DataLoader(full_data, batch_size=opt.batch_size, sampler=train_labeled_pseudo_resnet_sampler)

    return train_labeled_pseudo_vgg_loader, train_labeled_pseudo_inception_loader, train_labeled_pseudo_resnet_loader


def pseudolabeling(train_unlabeled_loader, vggsensor, inceptionsensor, resnetsensor):
    vggsensor.eval()
    inceptionsensor.eval()
    resnetsensor.eval()
    with torch.no_grad():
        for idx, (index, (im_w, _), labels) in enumerate(train_unlabeled_loader):
            im_w = im_w.cuda(non_blocking=True)
            # compute loss
            output_vgg = vggsensor(im_w)
            output_inception = inceptionsensor(im_w)
            output_resnet = resnetsensor(im_w)
            # update metric
            if idx == 0:
                index_set = index
                predict_set_vgg = output_vgg.detach().cpu().numpy()
                predict_set_inception = output_inception.detach().cpu().numpy()
                predict_set_resnet = output_resnet.detach().cpu().numpy()
            else:
                index_set = np.append(index_set, index, axis=0)
                predict_set_vgg = np.append(predict_set_vgg, output_vgg.detach().cpu().numpy(), axis=0)
                predict_set_inception = np.append(predict_set_inception, output_inception.detach().cpu().numpy(), axis=0)
                predict_set_resnet = np.append(predict_set_resnet, output_resnet.detach().cpu().numpy(), axis=0)
    # 验证predict_set_vgg、predict_set_inception和predict_set_resnet预测结果的一致性
    # 选择一致的样本
    index_vgg = np.where(np.argmax(predict_set_inception, axis=1) == np.argmax(predict_set_resnet, axis=1))[0]
    index_inception = np.where(np.argmax(predict_set_resnet, axis=1) == np.argmax(predict_set_vgg, axis=1))[0]
    index_resnet = np.where(np.argmax(predict_set_vgg, axis=1) == np.argmax(predict_set_inception, axis=1))[0]
    # 选择一致样本的伪标签
    aug_label_vgg = np.argmax(predict_set_inception[index_vgg], axis=1)
    aug_label_inception = np.argmax(predict_set_resnet[index_inception], axis=1)
    aug_label_resnet = np.argmax(predict_set_vgg[index_resnet], axis=1)
    # 生成伪标签数据集
    augdict_vgg = {'pseudo_idx': index_set[index_vgg], 'pseudo_label': aug_label_vgg}
    augdict_inception = {'pseudo_idx': index_set[index_inception], 'pseudo_label': aug_label_inception}
    augdict_resnet = {'pseudo_idx': index_set[index_resnet], 'pseudo_label': aug_label_resnet}
    return augdict_vgg, augdict_inception, augdict_resnet


def init_train(train_labeled_loader, vggsensor, inceptionsensor, resnetsensor, criterion, optimizer_vgg, optimizer_inception, optimizer_resnet, epoch, opt, tb):
    vggsensor.train()
    inceptionsensor.train()
    resnetsensor.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_vgg = AverageMeter()
    losses_inception = AverageMeter()
    losses_resnet = AverageMeter()
    top1_vgg = AverageMeter()
    top1_inception = AverageMeter()
    top1_resnet = AverageMeter()
    total_loss_vgg = 0
    total_loss_inception = 0
    total_loss_resnet = 0
    end = time.time()
    for idx, (_, (im_w, _), labels) in enumerate(train_labeled_loader):
        data_time.update(time.time() - end)
        im_w = im_w.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        # compute output
        # compute loss
        output_vgg = vggsensor(im_w)
        output_inception = inceptionsensor(im_w)
        output_resnet = resnetsensor(im_w)
        loss_vgg = criterion(output_vgg, labels)
        loss_inception = criterion(output_inception, labels)
        loss_resnet = criterion(output_resnet, labels)
        # update metric
        acc_vgg = accuracy(output_vgg, labels)
        top1_vgg.update(acc_vgg[0], labels.size(0))
        acc_inception = accuracy(output_inception, labels)
        top1_inception.update(acc_inception[0], labels.size(0))
        acc_resnet = accuracy(output_resnet, labels)
        top1_resnet.update(acc_resnet[0], labels.size(0))

        losses_vgg.update(loss_vgg.item(), labels.size(0))
        losses_inception.update(loss_inception.item(), labels.size(0))
        losses_resnet.update(loss_resnet.item(), labels.size(0))
        total_loss_vgg += loss_vgg.item()
        total_loss_inception += loss_inception.item()
        total_loss_resnet += loss_resnet.item()
        # SGD
        optimizer_vgg.zero_grad()
        loss_vgg.backward()
        optimizer_vgg.step()
        optimizer_inception.zero_grad()
        loss_inception.backward()
        optimizer_inception.step()
        optimizer_resnet.zero_grad()
        loss_resnet.backward()
        optimizer_resnet.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('InitTrain: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss-vgg {loss_vgg.val:.3f} ({loss_vgg.avg:.3f})\t'
                  'loss-inception {loss_inception.val:.3f} ({loss_inception.avg:.3f})\t'
                  'loss-resnet {loss_resnet.val:.3f} ({loss_resnet.avg:.3f})'.format(
                epoch, idx + 1, len(train_labeled_loader), batch_time=batch_time,
                data_time=data_time, loss_vgg=losses_vgg, loss_inception=losses_inception, loss_resnet=losses_resnet))
            sys.stdout.flush()
    # tensorboard
    tb.add_scalar("InitTrain/Acc-VGG", top1_vgg.avg, epoch)
    tb.add_scalar("InitTrain/Loss-VGG", total_loss_vgg, epoch)
    tb.add_scalar("InitTrain/Acc-Inception", top1_inception.avg, epoch)
    tb.add_scalar("InitTrain/Loss-Inception", total_loss_inception, epoch)
    tb.add_scalar("InitTrain/Acc-ResNet", top1_resnet.avg, epoch)
    tb.add_scalar("InitTrain/Loss-ResNet", total_loss_resnet, epoch)


def init_val(val_loader, vggsensor, inceptionsensor, resnetsensor, epoch, tb):
    vggsensor.eval()
    inceptionsensor.eval()
    resnetsensor.eval()
    top1_vgg = AverageMeter()
    top1_inception = AverageMeter()
    top1_resnet = AverageMeter()
    total_loss_vgg = 0
    total_loss_inception = 0
    total_loss_resnet = 0
    with torch.no_grad():
        for idx, (_, (im_w, _), labels) in enumerate(val_loader):
            im_w = im_w.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # compute loss
            output_vgg = vggsensor(im_w)
            output_inception = inceptionsensor(im_w)
            output_resnet = resnetsensor(im_w)
            # update metric
            acc_vgg = accuracy(output_vgg, labels)
            top1_vgg.update(acc_vgg[0], labels.size(0))
            acc_inception = accuracy(output_inception, labels)
            top1_inception.update(acc_inception[0], labels.size(0))
            acc_resnet = accuracy(output_resnet, labels)
            top1_resnet.update(acc_resnet[0], labels.size(0))

    # tensorboard
    tb.add_scalar("InitVal/Acc-VGG", top1_vgg.avg, epoch)
    tb.add_scalar("InitVal/Loss-VGG", total_loss_vgg, epoch)
    tb.add_scalar("InitVal/Acc-Inception", top1_inception.avg, epoch)
    tb.add_scalar("InitVal/Loss-Inception", total_loss_inception, epoch)
    tb.add_scalar("InitVal/Acc-ResNet", top1_resnet.avg, epoch)
    tb.add_scalar("InitVal/Loss-ResNet", total_loss_resnet, epoch)


def iter_train(train_loader, aug_dict, sensor, criterion, optimizer, epoch, opt):
    sensor.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    total_loss = 0
    end = time.time()
    aug_indics = aug_dict['pseudo_idx']
    aug_labels = aug_dict['pseudo_label']
    for idx, (indics, (im_w, _), labels) in enumerate(train_loader):
        for i in range(len(indics)):
            ins_idx = indics[i]
            if ins_idx in aug_indics:
                labels[i] = aug_labels[np.where(aug_indics == ins_idx)[0][0]]
                #labels[i] = aug_labels[aug_indics.index(ins_idx)]
        im_w = im_w.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        # compute loss
        output = sensor(im_w)
        loss = criterion(output, labels)
        losses.update(loss.item(), bsz)
        total_loss += loss.item()
        acc = accuracy(output, labels)
        top1.update(acc[0], labels.size(0))
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('IterTrain: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses))
            sys.stdout.flush()
    return top1.avg, total_loss


def iter_val(val_loader, vggsensor, inceptionsensor, resnetsensor, epoch, tb):
    vggsensor.eval()
    inceptionsensor.eval()
    resnetsensor.eval()
    top1_vgg = AverageMeter()
    top1_inception = AverageMeter()
    top1_resnet = AverageMeter()
    total_loss_vgg = 0
    total_loss_inception = 0
    total_loss_resnet = 0
    with torch.no_grad():
        for idx, (_, (im_w, _), labels) in enumerate(val_loader):
            im_w = im_w.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]
            # compute loss
            output_vgg = vggsensor(im_w)
            output_inception = inceptionsensor(im_w)
            output_resnet = resnetsensor(im_w)
            # update metric
            acc_vgg = accuracy(output_vgg, labels)
            top1_vgg.update(acc_vgg[0], labels.size(0))
            acc_inception = accuracy(output_inception, labels)
            top1_inception.update(acc_inception[0], labels.size(0))
            acc_resnet = accuracy(output_resnet, labels)
            top1_resnet.update(acc_resnet[0], labels.size(0))

    # tensorboard
    tb.add_scalar("IterVal/Acc-VGG", top1_vgg, epoch)
    tb.add_scalar("IterVal/Loss-VGG", total_loss_vgg, epoch)
    tb.add_scalar("IterVal/Acc-Inception", top1_inception, epoch)
    tb.add_scalar("IterVal/Loss-Inception", total_loss_inception, epoch)
    tb.add_scalar("IterVal/Acc-ResNet", top1_resnet, epoch)
    tb.add_scalar("IterVal/Loss-ResNet", total_loss_resnet, epoch)


def main():
    best_acc = 0
    opt = parse_option()
    tb = SummaryWriter(comment="iForestInitTraining_50")
    base_model = vgg()
    base_model = base_model.cuda()
    load_file = os.path.join('./save/iForest/basemodel', 'checkpoint_{epoch}.pth'.format(epoch=300))
    checkpoint = torch.load(load_file)
    base_model.load_state_dict(checkpoint['vggmodel'])
    train_data, train_loader, val_loader, test_loader = set_loader(opt)
    clean_data, clean_loader, noise_loader = clean_noise_division(base_model, train_data, opt)
    vggsensor, inceptionsensor, resnetsensor, criterion = set_trimodel()
    optimizer_vgg, optimizer_inception, optimizer_resnet = set_trioptimizer(opt, vggsensor, inceptionsensor, resnetsensor)
    if opt.epoch == 1:
        for epoch in range(opt.epoch, 101):
            # adjust_learning_rate(opt, optimizer, epoch)
            time1 = time.time()
            init_train(train_loader, vggsensor, inceptionsensor, resnetsensor, criterion, optimizer_vgg, optimizer_inception, optimizer_resnet, epoch, opt, tb)
            time2 = time.time()
            init_val(val_loader, vggsensor, inceptionsensor, resnetsensor, epoch, tb)
            if epoch % opt.save_freq == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'vggsensor': vggsensor.state_dict(),
                    'inceptionsensor': inceptionsensor.state_dict(),
                    'resnetsensor': resnetsensor.state_dict()
                }, opt.save_folder, epoch)
    else:
        load_file = os.path.join(opt.save_folder, 'checkpoint_{epoch}.pth'.format(epoch=opt.epoch))
        checkpoint = torch.load(load_file)
        vggsensor.load_state_dict(checkpoint['vggsensor'])
        inceptionsensor.load_state_dict(checkpoint['inceptionsensor'])
        resnetsensor.load_state_dict(checkpoint['resnetsensor'])
    for epoch in range(101, 401):
        # adjust_learning_rate(opt, optimizer, epoch)
        augdict_vgg, augdict_inception, augdict_resnet = pseudolabeling(noise_loader, vggsensor, inceptionsensor, resnetsensor)
        train_labeled_pseudo_vgg_loader, train_labeled_pseudo_inception_loader, train_labeled_pseudo_resnet_loader = pseudo_loader(
            opt, clean_data, augdict_vgg, augdict_inception, augdict_resnet)
        top1, loss = iter_train(train_labeled_pseudo_vgg_loader, augdict_vgg, vggsensor, criterion, optimizer_vgg, epoch, opt)
        tb.add_scalar("IterTrain/Acc-VGG", top1, epoch)
        tb.add_scalar("IterTrain/Loss-VGG", loss, epoch)
        top1, loss = iter_train(train_labeled_pseudo_inception_loader, augdict_inception, inceptionsensor, criterion, optimizer_inception, epoch, opt)
        tb.add_scalar("IterTrain/Acc-Inception", top1, epoch)
        tb.add_scalar("IterTrain/Loss-Inception", loss, epoch)
        top1, loss = iter_train(train_labeled_pseudo_resnet_loader, augdict_resnet, resnetsensor, criterion, optimizer_resnet, epoch, opt)
        tb.add_scalar("IterTrain/Acc-ResNet", top1, epoch)
        tb.add_scalar("IterTrain/Loss-ResNet", loss, epoch)

        if epoch % opt.save_freq == 0 and epoch >= 1:
            save_checkpoint({
                    'epoch': epoch,
                    'vggsensor': vggsensor.state_dict(),
                    'inceptionsensor': inceptionsensor.state_dict(),
                    'resnetsensor': resnetsensor.state_dict()
            }, opt.save_folder, epoch)


def save_checkpoint(state, save_folder, epoch):
    filename = os.path.join(save_folder, 'checkpoint_{epoch}.pth'.format(epoch=epoch))
    torch.save(state, filename)


if __name__ == '__main__':
    main()
