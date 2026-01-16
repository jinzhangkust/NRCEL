"""
Author: Dr. Jin Zhang
E-mail: j.zhang@kust.edu.cn
URL: https://jinzhangkust.github.io
Dept: Kunming University of Science and Technology
Created on 2025.06.06
"""
import random

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import time
import argparse
import numpy as np

from data import get_noised_froth_data, data4cls
from models.resnet_proj import resnet50
from models.inception_proj import inception
from models.ViT import vit_b_16_diy    # torchvision导入模型需将图片下采样到224*224

from losses import DisparityLoss
from mine import get_estimator, MINE_DV

from util import AverageMeter, accuracy


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=120, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers=4*num_GPU')
    parser.add_argument('--epoch', type=int, default=1, help='number of training epochs')
    parser.add_argument('--warmup_epoch', type=int, default=100, help='number of training epochs')
    parser.add_argument('--total_epochs', type=int, default=300, help='number of training epochs')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    # model dataset
    parser.add_argument('--model_name', type=str, default='NRCEL')
    # checkpoint
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    # create
    opt = parser.parse_args()
    # save
    opt.save_folder = os.path.join('./save', opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt


vggmodel = torchvision.models.vgg16(weights=None)
class VGGSensor(nn.Module):
    def __init__(self):
        super(VGGSensor, self).__init__()
        self.feature = vggmodel.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True))
        self.classifier = nn.Linear(512, 6)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
    def forward(self, x):
        x = self.feature(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        code = self.projector(x)
        out = self.classifier(code)
        return code, out

class InceptionSensor(nn.Module):
    def __init__(self):
        super(InceptionSensor, self).__init__()
        self.feature = inception()
        self.feature.add_module('pool', nn.AdaptiveAvgPool2d((1, 1)))
        self.projector = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True))
        self.classifier = nn.Linear(512, 6)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)

    def forward(self, x):
        x = self.feature(x).view(x.size(0), -1)
        code = self.projector(x)
        out = self.classifier(code)
        return code, out


class ResNetSensor(nn.Module):
    def __init__(self):
        super(ResNetSensor, self).__init__()
        self.feature = resnet50()
        self.feature.add_module('pool', nn.AdaptiveAvgPool2d((1, 1)))
        self.projector = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True))
        self.classifier = nn.Linear(512, 6)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)

    def forward(self, x):
        x = self.feature(x).view(x.size(0), -1)
        code = self.projector(x)
        out = self.classifier(code)
        return code, out


def set_model():
    vggsensor = VGGSensor()
    inceptionsensor = InceptionSensor()
    resnetsensor = ResNetSensor()
    ce_criterion = torch.nn.CrossEntropyLoss()
    disparity_criterion = DisparityLoss()
    if torch.cuda.is_available():
        vggsensor = vggsensor.cuda()
        inceptionsensor = inceptionsensor.cuda()
        resnetsensor = resnetsensor.cuda()
        ce_criterion = ce_criterion.cuda()
        disparity_criterion = disparity_criterion.cuda()
        cudnn.benchmark = True
    return vggsensor, inceptionsensor, resnetsensor, ce_criterion, disparity_criterion


def set_mine():
    mine_vgg = MINE_DV(1, 1)
    mine_inception = MINE_DV(1, 1)
    mine_resnet = MINE_DV(1, 1)
    optim_mine_vgg, _ = mine_vgg._configure_optimizers()
    optim_mine_inception, _ = mine_inception._configure_optimizers()
    optim_mine_resnet, _ = mine_resnet._configure_optimizers()
    return mine_vgg, mine_inception, mine_resnet, optim_mine_vgg, optim_mine_inception, optim_mine_resnet


def set_optimizer(opt, vggsensor, inceptionsensor, resnetsensor):
    optimizer_vgg = optim.Adam(vggsensor.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    optimizer_inception = optim.Adam(inceptionsensor.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    optimizer_resnet = optim.Adam(resnetsensor.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    return optimizer_vgg, optimizer_inception, optimizer_resnet


def set_loader(opt):
    full_data, train_data, val_data, test_data = get_noised_froth_data()
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_loader = DataLoader(val_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_loader = DataLoader(test_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    return full_data, train_loader, val_loader, test_loader


def clean_noise_division(opt, full_data, dict_inception, dict_resnet, dict_vgg):
    labeled_index_inception = list(dict_inception['clean_idx'])
    unlabeled_index_inception = list(dict_inception['noise_idx'])
    labeled_inception_sampler = torch.utils.data.SubsetRandomSampler(labeled_index_inception)
    unlabeled_inception_sampler = torch.utils.data.SubsetRandomSampler(unlabeled_index_inception)
    labeled_inception_loader = DataLoader(full_data, batch_size=opt.batch_size, sampler=labeled_inception_sampler)
    unlabeled_inception_loader = DataLoader(full_data, batch_size=opt.batch_size, sampler=unlabeled_inception_sampler)
    proto_inception = dict_inception['proto']

    labeled_index_resnet = list(dict_resnet['clean_idx'])
    unlabeled_index_resnet = list(dict_resnet['noise_idx'])
    labeled_resnet_sampler = torch.utils.data.SubsetRandomSampler(labeled_index_resnet)
    unlabeled_resnet_sampler = torch.utils.data.SubsetRandomSampler(unlabeled_index_resnet)
    labeled_resnet_loader = DataLoader(full_data, batch_size=opt.batch_size, sampler=labeled_resnet_sampler)
    unlabeled_resnet_loader = DataLoader(full_data, batch_size=opt.batch_size, sampler=unlabeled_resnet_sampler)
    proto_resnet = dict_resnet['proto']

    labeled_index_vgg = list(dict_vgg['clean_idx'])
    unlabeled_index_vgg = list(dict_vgg['noise_idx'])
    labeled_vgg_sampler = torch.utils.data.SubsetRandomSampler(labeled_index_vgg)
    unlabeled_vgg_sampler = torch.utils.data.SubsetRandomSampler(unlabeled_index_vgg)
    labeled_vgg_loader = DataLoader(full_data, batch_size=opt.batch_size, sampler=labeled_vgg_sampler)
    unlabeled_vgg_loader = DataLoader(full_data, batch_size=opt.batch_size, sampler=unlabeled_vgg_sampler)
    proto_vgg = dict_vgg['proto']

    return labeled_inception_loader, unlabeled_inception_loader, labeled_resnet_loader, unlabeled_resnet_loader, labeled_vgg_loader, unlabeled_vgg_loader, proto_inception, proto_resnet, proto_vgg


def warmup_train(train_loader, vgg, inception, resnet, ce_criterion, disparity_criterion, opti_vgg, opti_inception, opti_resnet, epoch, opt, tb):
    vgg.train()
    inception.train()
    resnet.train()
    top1_vgg = AverageMeter()
    top1_inception = AverageMeter()
    top1_resnet = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_celoss_vgg = 0
    total_disloss_vgg = 0
    total_celoss_inception = 0
    total_disloss_inception = 0
    total_celoss_resnet = 0
    total_disloss_resnet = 0
    end = time.time()
    for idx, (_, (im_w, im_s), labels) in enumerate(train_loader):
        im_w = im_w.cuda(non_blocking=True)
        im_s = im_s.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        # compute output
        code_w_vgg, out_w_vgg = vgg(im_w)
        code_s_vgg, out_s_vgg = vgg(im_s)
        code_w_inception, out_w_inception = inception(im_w)
        code_s_inception, out_s_inception = inception(im_s)
        code_w_resnet, out_w_resnet = resnet(im_w)
        code_s_resnet, out_s_resnet = resnet(im_s)
        # cross entropy loss
        ce_vgg = ce_criterion(out_w_vgg, labels)
        ce_inception = ce_criterion(out_w_inception, labels)
        ce_resnet = ce_criterion(out_w_resnet, labels)
        # disparity tri-training
        disparity_vgg = disparity_criterion(code_w_vgg, code_s_vgg, code_w_inception.detach(), code_w_resnet.detach())
        disparity_inception = disparity_criterion(code_w_inception, code_s_inception, code_w_vgg.detach(), code_w_resnet.detach())
        disparity_resnet = disparity_criterion(code_w_resnet, code_s_resnet, code_w_vgg.detach(), code_w_inception.detach())
        # update metric
        total_celoss_vgg += ce_vgg.item()
        total_disloss_vgg += disparity_vgg.item()
        total_celoss_inception += ce_inception.item()
        total_disloss_inception += disparity_inception.item()
        total_celoss_resnet += ce_resnet.item()
        total_disloss_resnet += disparity_resnet.item()
        acc_vit = accuracy(out_w_vgg, labels)
        top1_vgg.update(acc_vit[0], labels.size(0))
        acc_inception = accuracy(out_w_inception, labels)
        top1_inception.update(acc_inception[0], labels.size(0))
        acc_resnet = accuracy(out_w_resnet, labels)
        top1_resnet.update(acc_resnet[0], labels.size(0))
        # SGD
        vgg_losses = ce_vgg + 0.1 * disparity_vgg
        opti_vgg.zero_grad()
        vgg_losses.backward()
        opti_vgg.step()
        inception_losses = ce_inception + 0.1 * disparity_inception
        opti_inception.zero_grad()
        inception_losses.backward()
        opti_inception.step()
        resnet_losses = ce_resnet + 0.1 * disparity_resnet
        opti_resnet.zero_grad()
        resnet_losses.backward()
        opti_resnet.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    # tensorboard
    tb.add_scalar("WarmupTrain/Acc/VGG", top1_vgg.avg, epoch)
    tb.add_scalar("WarmupTrain/Acc/Inception", top1_inception.avg, epoch)
    tb.add_scalar("WarmupTrain/Acc/ResNet", top1_resnet.avg, epoch)
    tb.add_scalar("WarmupTrain/CELoss/VGG", total_celoss_vgg, epoch)
    tb.add_scalar("WarmupTrain/CELoss/Inception", total_celoss_inception, epoch)
    tb.add_scalar("WarmupTrain/CELoss/ResNet", total_celoss_resnet, epoch)
    tb.add_scalar("WarmupTrain/DisparityLoss/VGG", total_disloss_vgg, epoch)
    tb.add_scalar("WarmupTrain/DisparityLoss/Inception", total_disloss_inception, epoch)
    tb.add_scalar("WarmupTrain/DisparityLoss/ResNet", total_disloss_resnet, epoch)


def warmup_val(val_loader, vgg, inception, resnet, ce_criterion, disparity_criterion, epoch, tb):
    vgg.eval()
    inception.eval()
    resnet.eval()
    top1_vgg = AverageMeter()
    top1_inception = AverageMeter()
    top1_resnet = AverageMeter()
    total_celoss_vgg = 0
    total_disloss_vgg = 0
    total_celoss_inception = 0
    total_disloss_inception = 0
    total_celoss_resnet = 0
    total_disloss_resnet = 0
    with torch.no_grad():
        for idx, (_, (im_w, im_s), labels) in enumerate(val_loader):
            im_w = im_w.cuda(non_blocking=True)
            im_s = im_s.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # compute loss
            code_w_vgg, out_w_vgg = vgg(im_w)
            code_s_vgg, out_s_vgg = vgg(im_s)
            code_w_inception, out_w_inception = inception(im_w)
            code_s_inception, out_s_inception = inception(im_s)
            code_w_resnet, out_w_resnet = resnet(im_w)
            code_s_resnet, out_s_resnet = resnet(im_s)
            # cross entropy loss
            ce_vgg = ce_criterion(out_w_vgg, labels)
            ce_inception = ce_criterion(out_w_inception, labels)
            ce_resnet = ce_criterion(out_w_resnet, labels)
            # disparity tri-training
            disparity_vgg = disparity_criterion(code_w_vgg, code_s_vgg, code_w_inception, code_w_resnet)
            disparity_inception = disparity_criterion(code_w_inception, code_s_inception, code_w_vgg, code_w_resnet)
            disparity_resnet = disparity_criterion(code_w_resnet, code_s_resnet, code_w_vgg, code_w_inception)
            # update metric
            acc_vgg = accuracy(out_w_vgg, labels)
            top1_vgg.update(acc_vgg[0], labels.size(0))
            acc_inception = accuracy(out_w_inception, labels)
            top1_inception.update(acc_inception[0], labels.size(0))
            acc_resnet = accuracy(out_w_resnet, labels)
            top1_resnet.update(acc_resnet[0], labels.size(0))
            total_celoss_vgg += ce_vgg.item()
            total_disloss_vgg += disparity_vgg.item()
            total_celoss_inception += ce_inception.item()
            total_disloss_inception += disparity_inception.item()
            total_celoss_resnet += ce_resnet.item()
            total_disloss_resnet += disparity_resnet.item()
    # tensorboard
    tb.add_scalar("WarmupVal/Acc/VGG", top1_vgg.avg, epoch)
    tb.add_scalar("WarmupVal/Acc/Inception", top1_inception.avg, epoch)
    tb.add_scalar("WarmupVal/Acc/ResNet", top1_resnet.avg, epoch)
    tb.add_scalar("WarmupVal/CELoss/VGG", total_celoss_vgg, epoch)
    tb.add_scalar("WarmupVal/CELoss/Inception", total_celoss_inception, epoch)
    tb.add_scalar("WarmupVal/CELoss/ResNet", total_celoss_resnet, epoch)
    tb.add_scalar("WarmupVal/DisparityLoss/VGG", total_disloss_vgg, epoch)
    tb.add_scalar("WarmupVal/DisparityLoss/Inception", total_disloss_inception, epoch)
    tb.add_scalar("WarmupVal/DisparityLoss/ResNet", total_disloss_resnet, epoch)


def mutuallabeling(train_loader, vgg, inception, resnet):
    vgg.eval()
    inception.eval()
    resnet.eval()
    with torch.no_grad():
        for idx, (index, (im_w, _), labels) in enumerate(train_loader):
            im_w = im_w.cuda(non_blocking=True)
            #labels = labels.cuda(non_blocking=True)
            code_vgg, out_vgg = vgg(im_w)
            code_inception, out_inception = inception(im_w)
            code_resnet, out_resnet = resnet(im_w)
            out_vgg = np.argmax(out_vgg.detach().cpu().numpy(), axis=1)
            out_inception = np.argmax(out_inception.detach().cpu().numpy(), axis=1)
            out_resnet = np.argmax(out_resnet.detach().cpu().numpy(), axis=1)
            if idx == 0:
                index_set = index
                predict_set_vgg = out_vgg
                predict_set_inception = out_inception
                predict_set_resnet = out_resnet
                label_set = labels
                code_set_vgg = code_vgg.detach().cpu().numpy()
                code_set_inception = code_inception.detach().cpu().numpy()
                code_set_resnet = code_resnet.detach().cpu().numpy()
            else:
                index_set = np.append(index_set, index, axis=0)
                predict_set_vgg = np.append(predict_set_vgg, out_vgg, axis=0)
                predict_set_inception = np.append(predict_set_inception, out_inception, axis=0)
                predict_set_resnet = np.append(predict_set_resnet, out_resnet, axis=0)
                label_set = np.append(label_set, labels, axis=0)
                code_set_vgg = np.append(code_set_vgg, code_vgg.detach().cpu().numpy(), axis=0)
                code_set_inception = np.append(code_set_inception, code_inception.detach().cpu().numpy(), axis=0)
                code_set_resnet = np.append(code_set_resnet, code_resnet.detach().cpu().numpy(), axis=0)
    vgg_inception_match = (predict_set_vgg == predict_set_inception) & (predict_set_vgg == label_set)
    vgg_resnet_match = (predict_set_vgg == predict_set_resnet) & (predict_set_vgg == label_set)
    inception_resnet_match = (predict_set_inception == predict_set_resnet) & (predict_set_inception == label_set)
    clean_resnet_inception = np.where(vgg_inception_match)[0]
    noise_resnet_inception = np.where(~vgg_inception_match)[0]
    inception_proto = np.mean(code_set_inception[clean_resnet_inception], axis=0)
    dict_vgg = {'clean_idx': index_set[clean_resnet_inception], 'noise_idx': index_set[noise_resnet_inception], 'proto': inception_proto}
    clean_vgg_resnet = np.where(inception_resnet_match)[0]
    noise_vgg_resnet = np.where(~inception_resnet_match)[0]
    vgg_proto = np.mean(code_set_vgg[clean_vgg_resnet], axis=0)
    dict_inception = {'clean_idx': index_set[clean_vgg_resnet], 'noise_idx': index_set[noise_vgg_resnet], 'proto': vgg_proto}
    clean_vgg_inception = np.where(vgg_resnet_match)[0]
    noise_vgg_inception = np.where(~vgg_resnet_match)[0]
    resnet_proto = np.mean(code_set_resnet[clean_vgg_inception], axis=0)
    dict_resnet = {'clean_idx': index_set[clean_vgg_inception], 'noise_idx': index_set[noise_vgg_inception], 'proto': resnet_proto}
    return dict_vgg, dict_inception, dict_resnet

def iter_train(labeled_loader, unlabeled_loader, sensor, ce_criterion, optimizer, mine_model, mine_optimizer, opt):
    cosi = nn.CosineSimilarity(dim=1, eps=1e-6)
    sensor.train()
    batch_time = AverageMeter()
    top1 = AverageMeter()
    total_celoss = 0
    total_miloss = 0
    end = time.time()
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    labeled_loader_len = int(len(labeled_loader)/opt.batch_size)
    unlabeled_loader_len = int(len(unlabeled_loader)/opt.batch_size)
    train_iteration = labeled_loader_len if labeled_loader_len > unlabeled_loader_len else unlabeled_loader_len
    for batch_idx in range(train_iteration):
        try:
            _, (im_w_x, im_s_x), target_x = next(labeled_iter)
        except:
            labeled_iter = iter(labeled_loader)
            _, (im_w_x, im_s_x), target_x = next(labeled_iter)
        try:
            _, (im_w_u, im_s_u), _ = next(unlabeled_iter)
        except:
            unlabeled_iter = iter(unlabeled_loader)
            _, (im_w_u, im_s_u), _ = next(unlabeled_iter)
        if torch.cuda.is_available():
            im_w_x, im_s_x, target_x = im_w_x.cuda(non_blocking=True), im_s_x.cuda(non_blocking=True), target_x.cuda(non_blocking=True)
            im_w_u, im_s_u = im_w_u.cuda(non_blocking=True), im_s_u.cuda(non_blocking=True)

        bsz = target_x.shape[0]
        # learning with clean data
        code_x, out_x = sensor(im_w_x)
        ce_loss = ce_criterion(out_x, target_x)
        # learning with noise data
        code_w_u, out_w_u = sensor(im_w_u)
        code_s_u, out_s_u = sensor(im_w_u)
        shuffle_idx = torch.randperm(code_w_u.size(0))
        shuffled_code_u = code_s_u[shuffle_idx]
        shuffled_out_u = out_s_u[shuffle_idx]
        with torch.no_grad():
            feat_dist = torch.norm(code_w_u - shuffled_code_u, dim=1).unsqueeze(1)
            cls_simi = cosi(out_w_u, shuffled_out_u).unsqueeze(1)
        # updating the statistical network of MINE
        mi_loss = -mine_model.get_mi_bound(feat_dist, cls_simi)
        mine_optimizer.zero_grad()
        mi_loss.backward()
        mine_optimizer.step
        # calculating the upper bound of MI
        feat_dist = torch.norm(code_w_u - shuffled_code_u, dim=1).unsqueeze(1)
        cls_simi = cosi(out_w_u, shuffled_out_u).unsqueeze(1)
        mi_loss = -mine_model.get_mi_bound(feat_dist, cls_simi)
        losses = ce_loss + mi_loss
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        acc = accuracy(out_x, target_x)
        top1.update(acc[0], target_x.size(0))
        # SGD

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        total_celoss += ce_loss.item()
        total_miloss += mi_loss.item()
        return top1, total_celoss, total_miloss


def iter_val(val_loader, vggsensor, inceptionsensor, resnetsensor, epoch, tb):
    vggsensor.eval()
    inceptionsensor.eval()
    resnetsensor.eval()
    top1_vgg = AverageMeter()
    top1_inception = AverageMeter()
    top1_resnet = AverageMeter()
    with torch.no_grad():
        for idx, (_, (images, _), targets) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            # compute loss
            _, output_vgg = vggsensor(images)
            _, output_inception = inceptionsensor(images)
            _, output_resnet = resnetsensor(images)
            # update metric
            acc_vgg = accuracy(output_vgg , targets)
            top1_vgg.update(acc_vgg[0], targets.size(0))
            acc_inception = accuracy(output_inception, targets)
            top1_inception.update(acc_inception[0], targets.size(0))
            acc_resnet = accuracy(output_resnet, targets)
            top1_resnet.update(acc_resnet[0], targets.size(0))

    # tensorboard
    tb.add_scalar("IterVal/Acc/VGG", top1_vgg.avg, epoch)
    tb.add_scalar("IterVal/Acc/Inception", top1_inception.avg, epoch)
    tb.add_scalar("IterVal/Acc/ResNet", top1_resnet.avg, epoch)


def main():
    best_acc = 0
    opt = parse_option()
    tb = SummaryWriter(comment="NRCEL")
    full_data, train_loader, val_loader, test_loader = set_loader(opt)
    vggsensor, inceptionsensor, resnetsensor, ce_criterion, disparity_criterion = set_model()
    opti_vgg, opti_inception, opti_resnet = set_optimizer(opt, vggsensor, inceptionsensor, resnetsensor)
    mine_vgg, mine_inception, mine_resnet, optim_mine_vgg, optim_mine_inception, optim_mine_resnet = set_mine()
    for epoch in range(opt.epoch, opt.warmup_epoch+1):
        # adjust_learning_rate(opt, optimizer, epoch)
        warmup_train(train_loader, vggsensor, inceptionsensor, resnetsensor, ce_criterion, disparity_criterion, opti_vgg, opti_inception, opti_resnet, epoch, opt, tb)
        warmup_val(val_loader, vggsensor, inceptionsensor, resnetsensor, ce_criterion, disparity_criterion, epoch, tb)
        if epoch % opt.save_freq == 0 and epoch >= 1:
            save_checkpoint({
                    'epoch': epoch,
                    'vggsensor': vggsensor.state_dict(),
                    'inceptionsensor': inceptionsensor.state_dict(),
                    'resnetsensor': resnetsensor.state_dict(),
            }, opt.save_folder, epoch)


    for epoch in range(opt.warmup_epoch+1, opt.total_epochs + 1):
        # adjust_learning_rate(opt, optimizer, epoch)
        time1 = time.time()
        dict_vgg, dict_inception, dict_resnet = mutuallabeling(train_loader, vggsensor, inceptionsensor, resnetsensor)
        labeled_inception_loader, unlabeled_inception_loader, labeled_resnet_loader, unlabeled_resnet_loader, labeled_vgg_loader, unlabeled_vgg_loader, proto_inception, proto_resnet, proto_vgg = clean_noise_division(opt, full_data, dict_inception, dict_resnet, dict_vgg)
        acc, total_celoss, total_miloss = iter_train(labeled_vgg_loader, unlabeled_vgg_loader, resnetsensor, ce_criterion, opti_vgg, mine_vgg, optim_mine_vgg, opt)
        tb.add_scalar("IterTrain/Acc/VGG", acc.avg, epoch)
        tb.add_scalar("IterTrain/CELoss/VGG", total_celoss, epoch)
        tb.add_scalar("IterTrain/MILoss/VGG", total_miloss, epoch)
        acc, total_celoss, total_miloss = iter_train(labeled_inception_loader, unlabeled_inception_loader, inceptionsensor, ce_criterion, opti_inception, mine_inception, optim_mine_inception, opt)
        tb.add_scalar("IterTrain/Acc/Inception", acc.avg, epoch)
        tb.add_scalar("IterTrain/CELoss/Inception", total_celoss, epoch)
        tb.add_scalar("IterTrain/MILoss/Inception", total_miloss, epoch)
        acc, total_celoss, total_miloss = iter_train(labeled_resnet_loader, unlabeled_resnet_loader, resnetsensor, ce_criterion, opti_resnet, mine_resnet, optim_mine_resnet, opt)
        tb.add_scalar("IterTrain/Acc/ResNet", acc.avg, epoch)
        tb.add_scalar("IterTrain/CELoss/ResNet", total_celoss, epoch)
        tb.add_scalar("IterTrain/MILoss/ResNet", total_miloss, epoch)
        time2 = time.time()
        iter_val(val_loader, vggsensor, inceptionsensor, resnetsensor, epoch, tb)
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(opt.save_folder, 'VGGSensor_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(vggsensor.state_dict(), save_file)
            save_file = os.path.join(opt.save_folder, 'InceptionSensor_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(inceptionsensor.state_dict(), save_file)
            save_file = os.path.join(opt.save_folder, 'ResNetSensor_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(resnetsensor.state_dict(), save_file)
            save_file = os.path.join(opt.save_folder, 'VGGMINE_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(mine_vgg.state_dict(), save_file)
            save_file = os.path.join(opt.save_folder, 'InceptionMINE_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(mine_inception.state_dict(), save_file)
            save_file = os.path.join(opt.save_folder, 'ResNetMINE_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(mine_resnet.state_dict(), save_file)

def save_checkpoint(state, save_folder, epoch):
    filename = os.path.join(save_folder, 'checkpoint_{epoch}.pth'.format(epoch=epoch))
    torch.save(state, filename)
if __name__ == '__main__':
    main()
