#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, args):
        super(DNN, self).__init__()
        self.args = args
        self.fc = nn.Sequential(
                nn.Linear(args.feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(128),
                nn.Dropout(args.dropout),
                nn.Linear(128, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Dropout(args.dropout),
                nn.Linear(256, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout(args.dropout),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Dropout(args.dropout),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(128),
                nn.Dropout(args.dropout),
                nn.Linear(128, 2)
                )
        """self.linear1 = nn.Linear(args.feat_dim, 128, bias=True)
        self.linear2 = nn.Linear(128, 256, bias=True)
        self.linear3 = nn.Linear(256, 128, bias=True)
        self.linear4 = nn.Linear(128, args.output_dim, bias=True)"""

    def forward(self, x):
        x = self.fc(x)
        """x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))"""
        return x
        
class DNN_8(nn.Module):
    def __init__(self, args):
        super(DNN_8, self).__init__()
        self.args = args
        self.fc = nn.Sequential(
                nn.Linear(args.feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(128),
                nn.Dropout(args.dropout),
                nn.Linear(128, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Dropout(args.dropout),
                nn.Linear(256, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout(args.dropout),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(1024),
                nn.Dropout(args.dropout),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout(args.dropout),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Dropout(args.dropout),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(128),
                nn.Dropout(args.dropout),
                nn.Linear(128, 2)
                )
        """self.linear1 = nn.Linear(args.feat_dim, 128, bias=True)
        self.linear2 = nn.Linear(128, 256, bias=True)
        self.linear3 = nn.Linear(256, 128, bias=True)
        self.linear4 = nn.Linear(128, args.output_dim, bias=True)"""

    def forward(self, x):
        x = self.fc(x)
        """x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))"""
        return x
        
class DNN_tsne(nn.Module):
    def __init__(self):
        super(DNN_tsne, self).__init__()
        self.fc_1 = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(128),
                nn.Linear(128, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Linear(256, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Linear(256, 128))
                
        self.fc_2 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(128),
                nn.Linear(128, 2)
                )
        """self.linear1 = nn.Linear(args.feat_dim, 128, bias=True)
        self.linear2 = nn.Linear(128, 256, bias=True)
        self.linear3 = nn.Linear(256, 128, bias=True)
        self.linear4 = nn.Linear(128, args.output_dim, bias=True)"""

    def forward(self, x):
        feat = self.fc_1(x)
        x = self.fc_2(feat)
        """x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))"""
        return x, feat