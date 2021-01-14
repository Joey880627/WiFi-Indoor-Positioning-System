#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import subprocess
import pickle

import time
from wifi import Cell


class CollectorServer():
    def __init__(self, HOST = '0.0.0.0', PORT = 8000, search_time=2):
        self.search_time = search_time
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((HOST, PORT))
        self.s.listen(5)
        print('Server started at: %s:%s' % (HOST, PORT))
        
    def __get_one_data(self):
        success = False
        while not success:
            try:
                data = {}
                cells = Cell.all('wlan0')
                for cell in cells:
                    quality = cell.quality.split('/')
                    data[cell.address] = float(quality[0]) / float(quality[1])
                success = True
            except:
                pass
        return data
        
    def __get_data(self):
        data = {}
        start = time.time()
        while(time.time() - start <= self.search_time):
            data.update(self.__get_one_data())
        return data
        
    def start(self):
        try:
            while True:
                print('Wait for connection...')
                conn, addr = self.s.accept()
                print('Connected by ' + str(addr))
                i = 0
                try:
                    while True:
                        # data = self.__get_data()
                        data = self.__get_one_data()
                        # data_byte = pickle.dumps(data, -1)  #data serialized
                        data_byte = str(data).encode()  #data serialized
                        conn.send(data_byte)
                        i = i + 1
                        print('%d data sended, Dict length %d' %(i, len(data)))
                except ConnectionResetError:
                    pass
                except BrokenPipeError:
                    pass
                except Exception as e:
                    print(e)
                    pass
        except KeyboardInterrupt:
            self.s.close()
            print('\nServer closed')
        
if __name__ == '__main__':
    collector = CollectorServer()
    collector.start()