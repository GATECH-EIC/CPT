from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np

import os
import shutil
import argparse
import time
import logging
import math

import models
from modules.data import *


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name])
                     )


def parse_args():
    # hyper-parameters are from ResNet paper
    parser = argparse.ArgumentParser(
        description='PyTorch ImageNet training with CPT')
    parser.add_argument('--dir', help='annotate the working directory')
    parser.add_argument('--cmd', choices=['train', 'test'], default='train')
    parser.add_argument('--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: cifar10_resnet_38)')
    parser.add_argument('--dataset', '-d', type=str, default='imagenet',
                        choices=['cifar10', 'cifar100','imagenet'],
                        help='dataset choice')
    parser.add_argument('--datadir', default='/home/yf22/dataset', type=str,
                        help='path to dataset')
    parser.add_argument('--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')
    parser.add_argument('--epoch', default=120, type=int,
                        help='number of epochs (default: 90)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual iter number (useful on restarts)')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr_schedule', default='piecewise', type=str,
                        help='learning rate schedule')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print_freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to  latest checkpoint (default: None)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--step_ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--warm_up', action='store_true',
                        help='for n = 18, the model needs to warm up for 400 '
                             'iterations')
    parser.add_argument('--save_folder', default='save_checkpoints',
                        type=str,
                        help='folder to save the checkpoints')
    parser.add_argument('--eval_every', default=390, type=int,
                        help='evaluate model every (default: 1000) iterations')

    parser.add_argument('--num_bits',default=0,type=int,
                        help='num bits for weight and activation')
    parser.add_argument('--num_grad_bits',default=0,type=int,
                        help='num bits for gradient')
    parser.add_argument('--num_bits_schedule',default=None,type=int,nargs='*',
                        help='schedule for weight/act precision')
    parser.add_argument('--num_grad_bits_schedule',default=None,type=int,nargs='*',
                        help='schedule for grad precision')

    parser.add_argument('--is_cyclic_precision', action='store_true',
                        help='cyclic precision schedule')
    parser.add_argument('--cyclic_num_bits_schedule', default=None, type=int, nargs='*',
                        help='cyclic schedule for weight/act precision')
    parser.add_argument('--cyclic_num_grad_bits_schedule', default=None, type=int, nargs='*',
                        help='cyclic schedule for grad precision')
    parser.add_argument('--num_cyclic_period', default=32, type=int,
                        help='number of cyclic period for precision, same for weights/activation and gradients')
    parser.add_argument('--automatic_resume', action='store_true',
                        help='automatically resume from latest checkpoint')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    global save_path
    save_path = args.save_path = os.path.join(args.save_folder,
                                              "{}_num_bit_{}_{}_grad_bit_{}_{}_period_{}".format(args.arch,
                                                                                       args.cyclic_num_bits_schedule[0],
                                                                                       args.cyclic_num_bits_schedule[1],
                                                                                       args.cyclic_num_grad_bits_schedule[0],
                                                                                       args.cyclic_num_grad_bits_schedule[1],
                                                                                       args.num_cyclic_period))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # config logging file
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
    if os.path.exists(args.logger_file):
        os.remove(args.logger_file)
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    if args.cmd == 'train':
        logging.info('start training {}'.format(args.arch))
        run_training(args)

    elif args.cmd == 'test':
        logging.info('start evaluating {} with checkpoints from {}'.format(
            args.arch, args.resume))
        test_model(args)


def run_training(args):
    # create model
    training_loss = 0
    training_acc = 0

    model = models.__dict__[args.arch](args.pretrained)
    model = torch.nn.DataParallel(model).cuda()

    best_prec1 = 0
    best_epoch = 0

    if args.automatic_resume:
        latest_ckpt = os.path.join(args.save_path,'checkpoint_latest.pth.tar')
        if os.path.exists(latest_ckpt):
            checkpoint = torch.load(latest_ckpt)
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])

            logging.info('Automatically load latest checkpoint (epoch: {})'.format(checkpoint['epoch']))

        else:
            logging.info('No latest checkpoint. Train from scratch.')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])

            logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(
                args.resume, checkpoint['epoch']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False

    train_loader = prepare_train_data(dataset=args.dataset,
                                      datadir=args.datadir+'/train',
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)
    test_loader = prepare_test_data(dataset=args.dataset,
                                    datadir=args.datadir+'/val',
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    cr = AverageMeter()

    end = time.time()

    for _epoch in range(args.start_epoch, args.epoch):
        lr = adjust_learning_rate(args, optimizer, _epoch)

        print('Learning Rate:', lr)
        print('num bits:', args.num_bits, 'num grad bits:', args.num_grad_bits)

        for i, (input, target) in enumerate(train_loader):
            # measuring data loading time
            data_time.update(time.time() - end)

            cyclic_period = int((args.epoch * len(train_loader)) / args.num_cyclic_period)
            _iters = _epoch * len(train_loader) + i

            cyclic_adjust_precision(args, _iters, cyclic_period)

            model.train()

            fw_cost = args.num_bits*args.num_bits/32/32
            eb_cost = args.num_bits*args.num_grad_bits/32/32
            gc_cost = eb_cost
            cr.update((fw_cost+eb_cost+gc_cost)/3)

            target = target.squeeze().long().cuda()
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()

            # compute output
            output = model(input_var, args.num_bits, args.num_grad_bits)
            loss = criterion(output, target_var)
            training_loss += loss.item()

            # measure accuracy and record loss
            prec1, = accuracy(output.data, target, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            training_acc += prec1.item()

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print log
            if i % args.print_freq == 0:
                logging.info("Iter: [{0}][{1}/{2}]\t"
                             "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                             "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                             "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                             "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t".format(
                                _epoch,
                                i,
                                len(train_loader),
                                batch_time=batch_time,
                                data_time=data_time,
                                loss=losses,
                                top1=top1)
                )

        epoch = _epoch + 1
        epoch_loss = training_loss / len(train_loader)
        with torch.no_grad():
            prec1 = validate(args, test_loader, model, criterion, _epoch)

        logging.info('Epoch [{}] num_bits = {} num_grad_bits = {}'.format(epoch, args.num_bits, args.num_grad_bits))

        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = max(prec1, best_prec1)
            best_epoch = epoch

        logging.info("Current Best Prec@1: {} ".format(best_prec1))
        logging.info("Current Best Epoch: {}".format(best_epoch))

        checkpoint_path = os.path.join(args.save_path, 'checkpoint_{:05d}_{:.2f}.pth.tar'.format(_epoch, prec1))
        save_checkpoint({
            'epoch': _epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        },
            is_best, filename=checkpoint_path)
        shutil.copyfile(checkpoint_path, os.path.join(args.save_path,
                                                      'checkpoint_latest'
                                                      '.pth.tar'))


def validate(args, test_loader, model, criterion, _epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluation mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.squeeze().long().cuda()
        input_var = Variable(input, volatile=True).cuda()
        target_var = Variable(target, volatile=True).cuda()

        # compute output
        output = model(input_var, args.num_bits, args.num_grad_bits)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        top1.update(prec1.item(), input.size(0))
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % args.print_freq == 0) or (i == len(test_loader) - 1):
            logging.info(
                'Test: [{}/{}]\t'
                'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                    i, len(test_loader), batch_time=batch_time,
                    loss=losses, top1=top1
                )
            )

    logging.info('Epoch {} * Prec@1 {top1.avg:.3f}'.format(_epoch, top1=top1))
    return top1.avg


def test_model(args):
    # create model
    model = models.__dict__[args.arch](args.pretrained)
    model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(
                args.resume, checkpoint['epoch']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)
    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        prec1 = validate(args, test_loader, model, criterion, args.start_iter)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(save_path,
                                               'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def cyclic_adjust_precision(args, _iter, cyclic_period):
    if args.is_cyclic_precision:
        assert len(args.cyclic_num_bits_schedule) == 2
        assert len(args.cyclic_num_grad_bits_schedule) == 2

        num_bit_min = args.cyclic_num_bits_schedule[0]
        num_bit_max = args.cyclic_num_bits_schedule[1]

        num_grad_bit_min = args.cyclic_num_grad_bits_schedule[0]
        num_grad_bit_max = args.cyclic_num_grad_bits_schedule[1]

        args.num_bits = np.rint(num_bit_min +
                                0.5 * (num_bit_max - num_bit_min) *
                                (1 + np.cos(np.pi * ((_iter % cyclic_period) / cyclic_period) + np.pi)))
        args.num_grad_bits = np.rint(num_grad_bit_min +
                                     0.5 * (num_grad_bit_max - num_grad_bit_min) *
                                     (1 + np.cos(np.pi * ((_iter % cyclic_period) / cyclic_period) + np.pi)))
        # print(type(_iter))
        if _iter % 10 == 0:
            logging.info('Iter [{}] num_bits = {} num_grad_bits = {} cyclic precision'.format(_iter, args.num_bits,
                                                                                                  args.num_grad_bits))



def adjust_learning_rate(args, optimizer, _epoch):
    lr = args.lr * (0.1 ** (_epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()