#!/usr/bin/env python
# -*- coding: utf-8 -*-
from vpython import *
import numpy as np
import time

class Visualizer():
    def __init__(self):
        self.size = 0.1
        self.X = 9
        self.Y = 5
        
        self.scene = canvas(title="MD402", width=900, height=500, x=0, y=0, background=vec(0.95, 0.95, 0.95))
        self.scene.camera.pos = vec(self.X/2, self.Y/2, 5)
        self.scene.camera.axis = vec(0, 0, -5)
        scene.lights = []
        self.floor = box(pos=vec(self.X/2, self.Y/2, -self.size), size=vec(self.X, self.Y, 0.01), texture=textures.wood)
        self.gridLine = []
        for x in range(self.X+1):
            self.gridLine.append(box(pos=vec(x, self.Y/2, -self.size), size=vec(0.01, self.Y, 0.01), color=color.white))
        for y in range(self.Y+1):
            self.gridLine.append(box(pos=vec(self.X/2, y, -self.size), size=vec(self.X, 0.01, 0.01), color=color.white))
        self.rpi = box(pos=vec(0, 0, self.size), size=vec(0.2, 0.3, 0.01), color=vec(0.0, 0.75, 0.5))
        rate(1000)
    def update(self, x, y):
        self.rpi.pos = vec(x, y, self.size)

if __name__ == '__main__':
    v = Visualizer()
    while True:
        label = np.random.rand(2,) * np.array([9., 5.])
        print('Predict: (%.2f, %.2f)' %(label[0], label[1]))
        v.update(label[0], label[1])
        sleepTime = np.random.rand() * 0.5 + 1
        time.sleep(sleepTime)