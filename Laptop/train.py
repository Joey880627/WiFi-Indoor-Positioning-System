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
from config import Config
from dataset import WifiData
from model import DNN, DNN_8
from utils import IOStream
import time
from augmentation import jitter, drop

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp train.py checkpoints'+'/'+args.exp_name+'/'+'train.py.backup')
    os.system('cp test.py checkpoints' + '/' + args.exp_name + '/' + 'test.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp utils.py checkpoints' + '/' + args.exp_name + '/' + 'utils.py.backup')
    os.system('cp dataset.py checkpoints' + '/' + args.exp_name + '/' + 'dataset.py.backup')
    
def get_loader(args, config):
    dataset = WifiData(config, labeled = True)
    val_dataset = WifiData(config, valid = True)
    args.feat_dim = dataset.feat_dim()
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(args.val_split * dataset_size))
    print(f"Training on {dataset_size - split} data")
    # print(f"Validation on {split} data")
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)
    # validation_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler, drop_last=False)
    validation_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False)
    return train_loader, validation_loader

def distance(logits, label):
    return torch.norm((label - logits), dim=1).mean()

def train(args, config, io):
    train_loader, validation_loader = get_loader(args, config)
    device = torch.device("cuda" if args.cuda else "cpu")
    # print(len(train_loader), len(validation_loader))
    
    #Try to load models
    model = DNN(args).to(device)
    """if device == torch.device("cuda"):
        model = nn.DataParallel(model)"""
    if args.model_path != "":
        model.load_state_dict(torch.load(args.model_path))
    
    # for para in list(model.parameters())[:-5]:
    #     para.requires_grad=False
    # print(model)

    if args.use_sgd:
        # print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        # print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        """opt = optim.Adam([
        {'params': list(model.parameters())[:-1], 'lr':args.lr/50, 'weight_decay': 1e-4},
        {'params': list(model.parameters())[-1], 'lr':args.lr, 'weight_decay': 1e-4}
        ])
        """

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = nn.MSELoss()

    best_test_loss = 9999999.
    for epoch in range(args.epochs):
        startTime = time.time()
        
        ####################
        # Train
        ####################
        train_loss = 0.0
        train_dis = 0.0
        count = 0.0
        model.train()
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            data = drop(jitter(data, device), device)
            # data = jitter(data, device, delta=0.05)
            batch_size = data.shape[0]
            logits = model(data)
            loss = criterion(logits, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            dis = distance(logits, label)
            count += batch_size
            train_loss += loss.item() * batch_size
            train_dis += dis.item() * batch_size
        scheduler.step()
        outstr = 'Train %d, loss: %.6f, distance: %.6f' % (epoch, train_loss*1.0/count, train_dis*1.0/count)
        io.cprint(outstr)
        

        ####################
        # Evaluation
        ####################
        test_loss = 0.0
        test_dis = 0.0
        count = 0.0
        model.eval()
        with torch.no_grad():
            for data, label in validation_loader:
                data, label = data.to(device), label.to(device)
                batch_size = data.shape[0]
                logits = model(data)
                loss = criterion(logits, label)
                dis = distance(logits, label)
                count += batch_size
                test_loss += loss.item() * batch_size
                test_dis += dis.item() * batch_size
        outstr = 'Test %d, loss: %.6f, distance: %.6f' % (epoch, test_loss*1.0/count, test_dis*1.0/count)
        io.cprint(outstr)
        if test_loss <= best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
            torch.save(model, (config.root + config.model_path))
        io.cprint('Time: %.3f sec' %(time.time() - startTime))

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Wifi Indoor Positioning System')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=128, metavar='batch_size',
                        help='Size of batch')
    parser.add_argument('--n_address', type=int, default=32, metavar='n_address',
                        help='Size of address')
    parser.add_argument('--test_batch_size', type=int, default=64, metavar='batch_size',
                        help='Size of batch')
    parser.add_argument('--val_split', type=float, default=0.2, metavar='batch_size',
                        help='Validation portion')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()
    config = Config.from_json_file('config.json')
    args.feat_dim = config.n_address

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint('Using GPU')
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
        io.cprint('Using CPU')

    train(args, config, io)