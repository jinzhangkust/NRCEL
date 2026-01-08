"""
Author: Dr. Jin Zhang
E-mail: j.zhang@kust.edu.cn
URL: https://jinzhangkust.github.io
Dept: Kunming University of Science and Technology
Created on 2025.06.06

Pilot study for fully supervised learning with noisy labels
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

from dataset.data import get_noise_froth_data
from models.inception_proj import inception

from util import AverageMeter, accuracy


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=180, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers=4*num_GPU')
    parser.add_argument('--epoch', type=int, default=760, help='number of training epochs')
    parser.add_argument('--warmup_epoch', type=int, default=100, help='number of training epochs')
    parser.add_argument('--total_epochs', type=int, default=600, help='number of training epochs')
    parser.add_argument('--load_epoch', type=int, default=760, help='number of training epochs')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    # model dataset
    parser.add_argument('--model_name', type=str, default='Baseline')
    # checkpoint
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    # create
    opt = parser.parse_args()
    # save
    opt.save_folder = os.path.join('./save', opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt

#os.environ["CUDA_VISIBLE_DEVICES"] = '3'



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


def set_model():
    inceptionsensor = InceptionSensor()
    ce_criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        inceptionsensor = inceptionsensor.cuda()
        ce_criterion = ce_criterion.cuda()
        cudnn.benchmark = True
    return inceptionsensor, ce_criterion


def set_optimizer(opt, sensor):
    optimizer = optim.Adam(sensor.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    return optimizer


def set_loader(opt):
    train_data, val_data, test_data = get_noise_froth_data()
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    val_loader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    test_loader = DataLoader(test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    return train_loader, val_loader, test_loader


def train(train_loader, sensor, criterion, opti, epoch, opt, tb):
    sensor.train()
    top1 = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_loss = 0
    end = time.time()
    for idx, (_, (im_w, _), labels) in enumerate(train_loader):
        im_w = im_w.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        # compute output
        code, out = sensor(im_w)
        # cross entropy loss
        loss = criterion(out, labels)
        # update metric
        total_loss += loss.item()
        acc = accuracy(out, labels)
        top1.update(acc[0], labels.size(0))
        # SGD
        opti.zero_grad()
        loss.backward()
        opti.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    # tensorboard
    tb.add_scalar("Train/Acc", top1.avg, epoch)
    tb.add_scalar("Train/Loss", total_loss, epoch)


def val(val_loader, sensor, criterion, epoch, tb):
    sensor.eval()
    top1 = AverageMeter()
    total_loss = 0
    with torch.no_grad():
        for idx, (_, (im_w, _), labels) in enumerate(val_loader):
            im_w = im_w.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # compute loss
            code, out = sensor(im_w)
            # cross entropy loss
            loss = criterion(out, labels)
            # update metric
            acc = accuracy(out, labels)
            top1.update(acc[0], labels.size(0))
            total_loss += loss.item()
    # tensorboard
    tb.add_scalar("Val/Acc", top1.avg, epoch)
    tb.add_scalar("Val/Loss", total_loss, epoch)


def main():
    best_acc = 0
    opt = parse_option()
    tb = SummaryWriter(comment="Baseline")
    train_loader, val_loader, test_loader = set_loader(opt)
    sensor, criterion = set_model()
    opti = set_optimizer(opt, sensor)
    for epoch in range(1, opt.total_epochs + 1):
        # adjust_learning_rate(opt, optimizer, epoch)
        time1 = time.time()
        train(train_loader, sensor, criterion, opti, epoch, opt, tb)
        val(val_loader, sensor, criterion, epoch, tb)
        if epoch % opt.save_freq == 0:
            save_checkpoint({
                'epoch': epoch,
                'model': sensor.state_dict(),
                'optimizer': opti.state_dict(),
            }, opt.save_folder, epoch)

def save_checkpoint(state, save_folder, epoch):
    filename = os.path.join(save_folder, 'checkpoint_{epoch}.pth'.format(epoch=epoch))
    torch.save(state, filename)
if __name__ == '__main__':
    main()
