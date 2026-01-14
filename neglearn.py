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

import os
import sys
import time
import argparse
import numpy as np

from data import get_noised_froth_data

from util import AverageMeter, accuracy


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=200, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers=4*num_GPU')
    parser.add_argument('--epoch', type=int, default=400, help='number of training epochs')
    parser.add_argument('--warmup_epoch', type=int, default=100, help='number of training epochs')
    parser.add_argument('--total_epochs', type=int, default=600, help='number of training epochs')
    parser.add_argument('--load_epoch', type=int, default=300, help='number of training epochs')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    # model dataset
    parser.add_argument('--model_name', type=str, default='NegLearn_10')
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


class NLCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(NLCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, out, target):
        out_probs = torch.nn.functional.softmax(out, dim=1)  # compute softmax
        #print(f"out_probs: {out_probs[0, :]}")
        log_probs = torch.log(1-out_probs+1e-6)  # compute log(1-softmax)
        #log_probs = torch.nn.functional.log_softmax(out, dim=1)  # compute log softmax
        losses = -log_probs[:, target]  # gather the log probabilities with the target index
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses


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


def set_model():
    model = ResNetSensor()
    criterion = torch.nn.CrossEntropyLoss()
    nlcriterion = NLCrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        nlcriterion = nlcriterion.cuda()
        cudnn.benchmark = True
    return model, criterion, nlcriterion


def set_optimizer(opt, model):
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    return optimizer


def set_loader(opt):
    full_data, train_data, val_data, test_data = get_noised_froth_data()
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_loader = DataLoader(val_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_loader = DataLoader(test_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    return train_loader, val_loader, test_loader


def train_stage_one(train_loader, model, criterion, optimizer, epoch, tb):
    model.train()
    top1 = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_loss = 0
    end = time.time()
    for idx, (_, (im_w, _), target) in enumerate(train_loader):
        im_w = im_w.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        label = torch.zeros_like(target)
        for i, ref_label in enumerate(target):
            possible_negative = [c for c in range(6) if c != ref_label]
            neg_label = random.choice(possible_negative)
            neg_label = torch.tensor(neg_label).cuda()
            label[i] = neg_label
        label = label.cuda()
        # compute output
        out_w = model(im_w)
        # cross entropy loss
        loss = criterion(out_w, label)
        # update metric
        total_loss += loss.item()
        acc = accuracy(out_w, target)
        top1.update(acc[0], target.size(0))
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    # tensorboard
    tb.add_scalar("StageOne/Acc", top1.avg, epoch)
    tb.add_scalar("StageOne/Loss", total_loss, epoch)


def train_stage_two(train_loader, model, criterion, optimizer, epoch, tb):
    model.train()
    top1 = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_loss = 0
    end = time.time()
    for idx, (_, (im_w, _), target) in enumerate(train_loader):
        im_w = im_w.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        label = torch.zeros_like(target)
        for i, ref_label in enumerate(target):
            possible_negative = [c for c in range(6) if c != ref_label]
            neg_label = random.choice(possible_negative)
            neg_label = torch.tensor(neg_label).cuda()
            label[i] = neg_label
        label = label.cuda()
        # compute output
        out_w = model(im_w)
        # choose the top1 larger than 0.3
        out_w1 = torch.nn.functional.softmax(out_w, dim=1)
        out_w1 = out_w1.cpu().detach().numpy()
        mask = np.zeros(len(target), dtype=bool)
        for i, out in enumerate(out_w1):
            #print(f"out: {out.shape}    out.max(): {out.max()}")
            if out.max() >= 0.25:
                mask[i] = True
        if sum(mask) == 0:
            mask[0] = True
        #print(f"out_w: {out_w[mask]}    label: {label[mask]}")
        # calculate cross-entropy using masked data
        loss = criterion(out_w[mask], label[mask])
        # update metric
        total_loss += loss.item()
        acc = accuracy(out_w, target)
        top1.update(acc[0], target.size(0))
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    # tensorboard
    tb.add_scalar("StageTwo/Acc", top1.avg, epoch)
    tb.add_scalar("StageTwo/Loss", total_loss, epoch)


def train_stage_three(train_loader, model, criterion, optimizer, epoch, tb):
    model.train()
    top1 = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_loss = 0
    end = time.time()
    for idx, (_, (im_w, _), label) in enumerate(train_loader):
        im_w = im_w.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        # compute output
        out_w = model(im_w)
        # choose the top1 larger than 0.3
        outs = torch.nn.functional.softmax(out_w, dim=1)
        outs = outs.cpu().detach().numpy()
        mask = np.zeros(len(label), dtype=bool)
        for i, out in enumerate(outs):
            if out.max() >= 0.3:
                mask[i] = True
        if sum(mask) == 0:
            mask[0] = True
        # calculate cross-entropy using masked data
        loss = criterion(out_w[mask], label[mask])
        # update metric
        total_loss += loss.item()
        acc = accuracy(out_w, label)
        top1.update(acc[0], label.size(0))
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    # tensorboard
    tb.add_scalar("StageThree/Acc", top1.avg, epoch)
    tb.add_scalar("StageThree/Loss", total_loss, epoch)


def val(val_loader, model, criterion, epoch, tb):
    model.eval()
    top1 = AverageMeter()
    total_loss = 0
    with torch.no_grad():
        for idx, (_, (im_w, _), labels) in enumerate(val_loader):
            im_w = im_w.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # compute loss
            out_w = model(im_w)
            # cross entropy loss
            loss = criterion(out_w, labels)
            # update metric
            acc = accuracy(out_w, labels)
            top1.update(acc[0], labels.size(0))
            total_loss += loss.item()
    # tensorboard
    tb.add_scalar("NegLearn/Acc", top1.avg, epoch)
    tb.add_scalar("NegLearn/Loss", total_loss, epoch)


def main():
    best_acc = 0
    opt = parse_option()
    tb = SummaryWriter(comment="NegLearn_10")
    train_loader, val_loader, test_loader = set_loader(opt)
    model, criterion, nlcriterion = set_model()
    optimizer = set_optimizer(opt, model)
    no_pretrain = True
    if opt.epoch > 1:
        load_file = os.path.join(opt.save_folder, 'checkpoint_{epoch}.pth'.format(epoch=opt.epoch))
        checkpoint = torch.load(load_file)
        print(checkpoint.keys())
        model.load_state_dict(checkpoint['neglearn'])
    if opt.epoch < 100:
        for epoch in range(opt.epoch, 101):
            # adjust_learning_rate(opt, optimizer, epoch)
            train_stage_one(train_loader, model, nlcriterion, optimizer, epoch, tb)
            val(val_loader, model, criterion, epoch, tb)
            if epoch % opt.save_freq == 0 and epoch >= 1:
                save_checkpoint({
                        'epoch': epoch,
                        'neglearn': model.state_dict(),
                }, opt.save_folder, epoch)
        for epoch in range(101, 201):
            # adjust_learning_rate(opt, optimizer, epoch)
            train_stage_two(train_loader, model, nlcriterion, optimizer, epoch, tb)
            val(val_loader, model, criterion, epoch, tb)
            if epoch % opt.save_freq == 0 and epoch >= 1:
                save_checkpoint({
                        'epoch': epoch,
                        'neglearn': model.state_dict(),
                }, opt.save_folder, epoch)
    for epoch in range(opt.epoch+1, 601):
        # adjust_learning_rate(opt, optimizer, epoch)
        train_stage_three(train_loader, model, criterion, optimizer, epoch, tb)
        val(val_loader, model, criterion, epoch, tb)
        if epoch % opt.save_freq == 0 and epoch >= 1:
            save_checkpoint({
                    'epoch': epoch,
                    'neglearn': model.state_dict(),
            }, opt.save_folder, epoch)


def save_checkpoint(state, save_folder, epoch):
    filename = os.path.join(save_folder, 'checkpoint_{epoch}.pth'.format(epoch=epoch))
    torch.save(state, filename)


if __name__ == '__main__':
    main()
