import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import WifiData
import time
from config import Config

def jitter(data, device, delta=0.05):
    """
    Randomly jitter the Wifi data
    data = data + uniform(-delta, delta)
    """
    with torch.no_grad():
        data += torch.FloatTensor(data.shape).uniform_(-delta, delta).to(device)
        data = torch.clamp(data, 0.0, 1.0)
        return data
    
def drop(data, device, drop_rate=0.2):
    """
    Randomly drop some Wifi signal
    data = data * mask
    """
    with torch.no_grad():
        mask = (torch.FloatTensor(data.shape).uniform_() > drop_rate).to(device)
        data = data * mask
        return data
    

if __name__ == '__main__':
    config = Config.from_json_file('config.json')
    device = 'cpu'
    data_loader = DataLoader(WifiData(config), batch_size=32, shuffle=False, drop_last=True)
    for data, label in data_loader:
        data, label = data.to(device), label.to(device)
        
        startTime = time.time()
        jitter_data = jitter(data, device)
        print("Jitter Wifi data time:", time.time() - startTime)
        
        startTime = time.time()
        drop_data = drop(data, device)
        print("Drop Wifi data time: %.6f" %(time.time() - startTime))
        
        break