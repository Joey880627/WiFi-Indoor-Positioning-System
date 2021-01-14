#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import socket
import pickle
import json
import ast
import threading
import time
import datetime
from config import Config
import sys

class ClientThread(threading.Thread):
    def __init__(self, host='192.168.137.183', port=8000, label=None):
        super(ClientThread, self).__init__()
        self.data = []
        self.host = host
        self.port = port
        self.label = label
        self.do_run = True
        self.data = []
    def __start_socket(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
            
    def __stop_socket(self):
        self.socket.close()
    def run(self):
        self.__start_socket()
        t = threading.currentThread()
        while getattr(t, "do_run", True):
            try:
                outdata = self.socket.recv(16384)
                # data = pickle.loads(data) #data loaded.
                outdata = ast.literal_eval(outdata.decode())
                data = {}
                if self.label:
                    for key, value in outdata.items():
                        outdata[key] = float(value) / 70
                    data['Feature'] = outdata
                    data['Label'] = self.label
                else:
                    for key, value in outdata.items():
                        outdata[key] = float(value) / 70
                    data['Feature'] = outdata
                self.data.append(data)
                print('Recieve a dict with length %d from %s' %(len(outdata), self.host))
            except Exception as e:
                print(e)
        self.__stop_socket()

class CollectorClient():
    def __init__(self, config, labeled=True):
        self.config = config
        if not os.path.exists(config.root):
            os.makedirs(config.root)
        if not os.path.exists(config.root + config.labeled_path):
            os.makedirs(config.root + config.labeled_path)
        if not os.path.exists(config.root + config.unlabeled_path):
            os.makedirs(config.root + config.unlabeled_path)
        self.labeled = labeled
        datetime_dt = datetime.datetime.today()
        datetime_str = datetime_dt.strftime("%Y%m%d_%H%M%S")
        if self.labeled:
            self.path = config.root + config.labeled_path + 'data' + datetime_str + '.pkl'
        else:
            self.path = config.root + config.unlabeled_path + 'data' + datetime_str + '.pkl'
        self.__ClearData()
    def __ClearData(self):
        self.data = []
    def start(self):
        if self.labeled:
            while True:
                print("************************************************")
                label = input("Please input the label of the data\nUse space to split the xy values (Ex:3 5)\nInput q to quit the proccess: ")
                print("************************************************")
                if (label=='q') or (label=='Q'):
                    break
                try:
                    label = label.split(' ')
                    label[0] = int(label[0])
                    label[1] = int(label[1])
                except:
                    raise RuntimeError('Invalid input (Input must be two integer values split with a space)')
                print('Label:', label)
                collectors = []
                for i, HOST in enumerate(self.config.hosts):
                    collectors.append(ClientThread(host=HOST, port=self.config.port, label=label))
                    collectors[i].start()
                stop = input('Type in anything to stop reading\n')
                for collector in collectors:
                    collector.do_run = False
                    collector.join()
                    self.data = self.data + collector.data
                self.SaveData()
        else:
            print('Unlabeled data')
            collectors = []
            for i, HOST in enumerate(self.config.hosts):
                collector = ClientThread(host=HOST, port=self.config.port, label=None)
                collectors.append(collector)
                collectors[i].start()
            stop = input('Type in anything to stop the program\n')
            for collector in collectors:
                collector.do_run = False
                collector.join()
                self.data = self.data + collector.data
    def SaveData(self):
        print('\nDump data to %s' %self.path)
        with open(self.path, 'ab') as fout:
            pickle.dump(self.data, fout)
        
if __name__ == '__main__':
    
    config = Config.from_json_file('config.json')
    try:
        if sys.argv[1] == 'labeled':
            labeled = True
        elif sys.argv[1] == 'unlabeled':
            labeled = False
    except:
        labeled = True
    collector = CollectorClient(config, labeled=labeled)
    collector.start()
    collector.SaveData()