#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from config import Config
from dataset import WifiData
from model import DNN
from utils import IOStream
from semi_utils import sigmoid_rampup, get_current_consistency_weight, update_ema_variables
from itertools import repeat, cycle
import time
from augmentation import jitter, drop

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp train_semi.py checkpoints'+'/'+args.exp_name+'/'+'train_semi.py.backup')
    os.system('cp test.py checkpoints' + '/' + args.exp_name + '/' + 'test.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp utils.py checkpoints' + '/' + args.exp_name + '/' + 'utils.py.backup')
    os.system('cp dataset.py checkpoints' + '/' + args.exp_name + '/' + 'dataset.py.backup')
    os.system('cp collect.py checkpoints' + '/' + args.exp_name + '/' + 'collect.py.backup')
    
def get_loader(args, config):
    dataset = WifiData(config, labeled=True)
    args.feat_dim = dataset.feat_dim()
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(args.val_split * dataset_size))
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)
    validation_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler, drop_last=False)
    unlabelled_dataset = WifiData(config, labeled = False)
    unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    return train_loader, validation_loader, unlabelled_loader



def train(args, config, io):
    train_loader, validation_loader, unlabelled_loader = get_loader(args, config)

    device = torch.device("cuda" if args.cuda else "cpu")
    
    #Try to load models
    model = DNN(args).to(device)
    ema_model = DNN(args).to(device)
    for param in ema_model.parameters():
        param.detach_()
    if device == torch.device("cuda"):
        model = nn.DataParallel(model)
        ema_model = nn.DataParallel(ema_model)
    if args.model_path != "":
        model.load_state_dict(torch.load(args.model_path))
        ema_model.load_state_dict(torch.load(args.model_path))

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = nn.MSELoss()
    consistency_criterion = nn.MSELoss()

    best_test_loss = 9999999.
    global_step = 0
    for epoch in range(args.epochs):
        startTime = time.time()
        
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        ema_model.train()
        i = -1
        for (data, label), (u, _) in zip(cycle(train_loader), unlabelled_loader):
            i = i + 1
            if data.shape[0]!= u.shape[0]:
                bt_size = np.minimum(data.shape[0], u.shape[0])
                data = data[0:bt_size]
                label = label[0:bt_size]
                u = u[0:bt_size]
            data, label, u = data.to(device), label.to(device), u.to(device)
            batch_size = data.shape[0]
            logits = model(data)
            class_loss = criterion(logits, label)
            
            u_student = jitter(u, device)
            u_teacher = jitter(u, device)
            logits_unlabeled = model(u_student)
            ema_logits_unlabeled = ema_model(u_teacher)
            ema_logits_unlabeled = Variable(ema_logits_unlabeled.detach().data, requires_grad=False)
            consistency_loss = consistency_criterion(logits_unlabeled, ema_logits_unlabeled)
            if epoch < args.consistency_rampup_starts:
                consistency_weight = 0.0
            else:
                consistency_weight = get_current_consistency_weight(args, args.final_consistency, epoch, i, len(unlabelled_loader))
            
            consistency_loss = consistency_weight * consistency_loss
            loss = class_loss + consistency_loss
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            global_step += 1
            # print(global_step)
            update_ema_variables(model, ema_model, args.ema_decay, global_step)
            
            count += batch_size
            train_loss += loss.item() * batch_size
        scheduler.step()
        outstr = 'Train %d, loss: %.6f' % (epoch, train_loss*1.0/count)
        io.cprint(outstr)
        

        ####################
        # Evaluation
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        ema_model.eval()
        for data, label in validation_loader:
            data, label = data.to(device), label.to(device)
            batch_size = data.shape[0]
            logits = ema_model(data)
            loss = criterion(logits, label)
            count += batch_size
            test_loss += loss.item() * batch_size
        outstr = 'Test %d, loss: %.6f' % (epoch, test_loss*1.0/count)
        io.cprint(outstr)
        if test_loss <= best_test_loss:
            best_test_loss = test_loss
            torch.save(ema_model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
            torch.save(ema_model, (config.root + config.model_path))
        io.cprint('Time: %.3f sec' %(time.time() - startTime))

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Wifi Indoor Positioning System')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch')
    parser.add_argument('--n_address', type=int, default=32, metavar='n_address',
                        help='Size of address')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch')
    parser.add_argument('--val_split', type=float, default=0.2, metavar='batch_size',
                        help='Validation portion')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--feat_dim', type=int, default=128, metavar='N',
                        help='Dimension of input')
    parser.add_argument('--output_dim', type=int, default=2, metavar='N',
                        help='Dimension of output')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--consistency_type', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--ema_decay', default=0.999, type=float, metavar='ALPHA',
                    help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--final_consistency', default=1.0, type=float,
                        help='consistency coeff for mixup usup loss')
    parser.add_argument('--consistency_rampup_starts', default=0, type=int, metavar='EPOCHS',
                        help='epoch at which consistency loss ramp-up starts')
    parser.add_argument('--consistency_rampup_ends', default=25, type=int, metavar='EPOCHS',
                        help='epoch at which consistency loss ramp-up ends')
    args = parser.parse_args()
    config = Config.from_json_file('config.json')

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint('Using GPU')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    train(args, config, io)