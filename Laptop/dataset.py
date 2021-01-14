#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import numpy as np
import json
import pickle
from torch.utils.data import Dataset
import torch
from collections import Counter
from config import Config
import os


class WifiData(Dataset):
    """
    Wifi data class
    n_address: The address numbers we select
    data: (batch_size, n_address)
    labels: (batch_size, 2) # xy coordinates
    """
    def __init__(self, config, valid=False, test=False, labeled=True):
        self.config = config
        self.labeled = labeled
        self.valid = valid
        self.test = test
        self.raw = []
        self.NO_LABEL = -1
        if self.valid:
            self.data_path = os.path.join(self.config.root, self.config.validation_path)
            self.save_path = os.path.join(self.config.root, self.config.validation_path, 'data.pkl')
        elif test:
            self.data_path = os.path.join(self.config.root, self.config.test_path)
            self.save_path = os.path.join(self.config.root, self.config.test_path, 'data.pkl')
        else:
            if self.labeled:
                self.data_path = os.path.join(self.config.root, self.config.labeled_path)
                self.save_path = os.path.join(self.config.root, self.config.labeled_path, 'data.pkl')
            else:
                self.data_path = os.path.join(self.config.root, self.config.unlabeled_path)
                self.save_path = os.path.join(self.config.root, self.config.unlabeled_path, 'data.pkl')
        
        self.__load_raw()
        if self.valid:
            self.addr_to_index = Counter(json.load(open(os.path.join(self.config.root, self.config.map_path))))
            self.__make_data()
        if self.test:
            # Load an existing dict
            self.addr_to_index = Counter(json.load(open(os.path.join(self.config.root, self.config.map_path))))
            self.__make_unlabeled_data()
        elif self.labeled:
            # Define a dict from address to index
            self.addr_to_index = self.__createmap()
            with open(os.path.join(self.config.root, self.config.map_path), 'w') as fout:
                json.dump(self.addr_to_index, fout)
            self.__make_data()
        else:
            # Load an existing dict
            self.addr_to_index = Counter(json.load(open(os.path.join(self.config.root, self.config.map_path))))
            self.__make_unlabeled_data()
        
    def __load_raw(self):
        for path in glob.glob(os.path.join(self.data_path, 'data*.pkl')):
            self.raw = self.raw + pickle.load(open(path, 'rb'))
    
    def __createmap(self):
        addresses = []
        addr_to_index = {}
        for i, data in enumerate(self.raw):
            addresses += list(data['Feature'].keys())
        counter = Counter(addresses).most_common(self.config.n_address)
        addr_to_index = Counter({addr:i+1 for i, (addr, _) in enumerate(counter)})
        return addr_to_index
        
    def __make_data(self):
        data_num = len(self.raw)
        feat_dim = len(self.addr_to_index)
        self.data = np.zeros((data_num, feat_dim), dtype=np.float32)
        self.label = np.zeros((data_num, 2), dtype=np.float32)
        
        for i, data in enumerate(self.raw):
            for j, address in enumerate(list(data['Feature'].keys())):
                index = self.addr_to_index[address]
                quality = data['Feature'][address]
                if not index:
                    continue
                else:
                    self.data[i][index-1] = quality
            self.label[i] = np.array(data['Label'])
            
    def __make_unlabeled_data(self):
        data_num = len(self.raw)
        feat_dim = len(self.addr_to_index)
        self.data = np.zeros((data_num, feat_dim), dtype=np.float32)
        
        for i, data in enumerate(self.raw):
            for j, address in enumerate(list(data['Feature'].keys())):
                index = self.addr_to_index[address]
                quality = data['Feature'][address]
                if not index:
                    continue
                else:
                    self.data[i][index-1] = quality

    def __getitem__(self, item):
        if self.test:
            return self.data[item], self.NO_LABEL
        else:
            if self.labeled:
                return self.data[item], self.label[item]
            else:
                return self.data[item], self.NO_LABEL

    def __len__(self):
        return self.data.shape[0]
        
    def feat_dim(self):
        return len(self.addr_to_index)
    
    def SaveData(self):
        print('Dump data to %s' %self.save_path)
        with open(self.save_path, 'wb') as fout:
            pickle.dump(self.raw, fout)
        
if __name__ == '__main__':
    config = Config.from_json_file('config.json')
    
    labeled_wifiData = WifiData(config, labeled = True)
    data, label = labeled_wifiData[0]
    print('Labeled Data Num:', len(labeled_wifiData))
    print("Labeled Data shape:", data.shape)
    print("Label shape:", label.shape)
    # labeled_wifiData.SaveData()
    
    """
    label_list = []
    for data, label in labeled_wifiData:
        label_list.append(label)
    print(np.unique(np.array(label_list), axis=0))"""
    
    """try:
        unlabeled_wifiData = WifiData(config, labeled = False)
        data, _ = unlabeled_wifiData[0]
        print('Unlabeled Data Num:', len(unlabeled_wifiData))
        print("Unlabeled Data shape:", data.shape)
        unlabeled_wifiData.SaveData()
    except Exception as e:
        print("No unlabeled data", e)"""
    