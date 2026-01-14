# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import re
import argparse
import os
import shutil
import time
import math
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

import losses
from util import *
from data import get_noised_froth_data
from models.resnet_proj import resnet50
from torch.utils.tensorboard import SummaryWriter


args = None
best_prec1 = 0
global_step = 0

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=301, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=300, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--labeled-batch-size', default=None, type=int, metavar='N', help="labeled examples per minibatch (default: no constrain)")
parser.add_argument('--num_workers', type=int, default=4, help='num_workers=4*num_GPU')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='max learning rate')
parser.add_argument('--initial-lr', default=0.0, type=float, metavar='LR', help='initial learning rate when using linear rampup')
parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS', help='length of learning rate rampup in the beginning')
parser.add_argument('--lr-rampdown-epochs', default=None, type=int, metavar='EPOCHS', help='length of learning rate cosine rampdown (>= length of training)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--nesterov', default=False, type=bool,help='use nesterov momentum', metavar='BOOL')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA', help='ema variable decay rate (default: 0.999)')
parser.add_argument('--consistency', default=1, type=float, metavar='WEIGHT', help='use consistency loss with given weight (default: None)')
parser.add_argument('--consistency-type', default="mse", type=str, metavar='TYPE', choices=['mse', 'kl'], help='consistency loss type to use')
parser.add_argument('--consistency-rampup', default=30, type=int, metavar='EPOCHS', help='length of the consistency loss ramp-up')
parser.add_argument('--logit-distance-cost', default=-1, type=float, metavar='WEIGHT', help='let the student model have two outputs and use an MSE loss between the logits with the given weight (default: only have one output)')
parser.add_argument('--checkpoint-epochs', default=10, type=int, metavar='EPOCHS', help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
parser.add_argument('--evaluation-epochs', default=1, type=int, metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=0, type=int, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=bool, help='evaluate model on evaluation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


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


def create_model(ema=False):
    model = ResNetSensor().cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model
    
    
def main():
    global global_step
    global best_prec1

    checkpoint_path = "./save/MT_50"
    writer = SummaryWriter(comment="MeanTeacher_50")
    
    if args.resume:
        args.resume = "./save/MT/checkpoint_{}.ckpt".format(args.resume)
    
    train_loader, val_loader, test_loader = set_loader(args)
    
    model = create_model()
    ema_model = create_model(ema=True)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    if args.evaluate:
        validate(test_loader, model, global_step, args.start_epoch)
        validate(test_loader, ema_model, global_step, args.start_epoch)
        return

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # train for one epoch
        class_loss, cons_loss, top1 = train(train_loader, model, ema_model, optimizer, epoch)

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            prec1 = validate(val_loader, model, global_step, epoch + 1)
            ema_prec1 = validate(val_loader, ema_model, global_step, epoch + 1)
            is_best = ema_prec1 > best_prec1
            best_prec1 = max(ema_prec1, best_prec1)
        else:
            is_best = False

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint_path, epoch + 1)
            
        writer.add_scalar('train/class_loss', class_loss, epoch)
        writer.add_scalar('train/cons_loss', cons_loss, epoch)
        writer.add_scalar('train/train_top1', top1, epoch)
        writer.add_scalar('test_top1', prec1, epoch)


def set_loader(opt):
    train_data, val_data, test_data = get_noised_froth_data()
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_loader = DataLoader(val_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_loader = DataLoader(test_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    return train_loader, val_loader, test_loader


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(train_loader, model, ema_model, optimizer, epoch):
    global global_step

    class_criterion = nn.CrossEntropyLoss().cuda()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    residual_logit_criterion = losses.symmetric_mse_loss

    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()

    end = time.time()

    model.train()
    for idx, (_, (im_w, im_s), target) in enumerate(train_loader):

        bsz = im_w.size(0)

        # Transform label to one-hot
        #targets_x = torch.zeros(batch_size, 6).scatter_(1, targets_x.view(-1,1).long(), 1)

        if torch.cuda.is_available():
            im_w = im_w.cuda()
            im_s = im_s.cuda()
            target = target.cuda()
            
        adjust_learning_rate(optimizer, epoch, idx, len(train_loader))
        meters.update('lr', optimizer.param_groups[0]['lr'])
        model_out = model(im_w)
        ema_model_out = ema_model(im_s)

        class_loss = class_criterion(model_out, target)
        meters.update('class_loss', class_loss)
        
        consistency_weight = get_current_consistency_weight(epoch)
        meters.update('cons_weight', consistency_weight)
        consistency_loss = consistency_weight * consistency_criterion(model_out, ema_model_out) / bsz
        meters.update('cons_loss', consistency_loss)

        loss = class_loss + 0.2*consistency_loss
        meters.update('loss', loss)

        prec1, prec5 = accuracy(model_out, target, topk=(1, 5))
        meters.update('top1', prec1[0], bsz)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(model, ema_model, args.ema_decay, global_step)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()
            
    return meters['class_loss'].avg, meters['cons_loss'].avg, meters['top1'].avg

        


def validate(eval_loader, model, global_step, epoch):
    class_criterion = nn.CrossEntropyLoss().cuda()
    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (_, (im_w, im_s), target) in enumerate(eval_loader):
        with torch.no_grad():
            im_w = im_w.cuda()
            target = target.cuda()

            minibatch_size = im_w.size(0)

            # compute output
            output = model(im_w)
            class_loss = class_criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            meters.update('class_loss', class_loss, minibatch_size)
            meters.update('top1', prec1[0], minibatch_size)

            # measure elapsed time
            meters.update('batch_time', time.time() - end)
            end = time.time()

    return meters['top1'].avg


def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint_{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr
    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= cosine_rampdown(epoch, args.lr_rampdown_epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        lr =  1.0
    else:
        lr = current / rampup_length
    
    #print (lr)
    return lr

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            #print(f"k: {k}   correct: {correct[:k].size()}")
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
