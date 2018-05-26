#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
Simulation for UAV Mapping
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
import numpy as np
import cv2
import sys
import os

import RRT
"""
class UAVSimulation(object):

    def __init__(self):
        self.DISPLAY = True
        self.vehicle_radius = 3
        self.display_size = (self.environment_size[1],self.environment_size[0],3)
        self.environment_size = (1000,1500,500)

        self.display = np.zeros(self.display_size,dtype=np.uint8)
        self.enironment = np.array(self.environment_size,dtype=np.uint8)
        self.init_simulation()

    def update_vehicle(self,x,y,z):
        self.previous_location = self.vehicle_location
        self.vehicle_location = (x,y,z)

    def draw_update(self):
        cv2.


    def init_simulation(self):
        while True:
            self.update_vehicle()
            self.update_obstacles()
            self.create_path()
            if self.DISPLAY:
                self.draw_update()
"""
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = plt.axes(projection='3d')
#ax = plt.axes(projection='3d')

# Data for a three-dimensional line
#zline = np.linspace(0, 15, 1000)
#xline = np.sin(zline)
#yline = np.cos(zline)
#ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
#zdata = 15 * np.random.random(100)
#xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
#ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
#ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

# plt.axis([0, 10, 0, 1])

obstacleList = [
        #(5, 5, 1),
        #(3, 6, 2),
        #(3, 8, 2),
        #(3, 10, 2),
        (7, 5, 2),
        (9, 5, 2)
    ]  # [x,y,size]

waypoints = [
    
    (0,0,0),
    (10,10,10),
    (-10,-10,2)
]


current_height = 0
for i in range(100):
    plt.cla()
    xline = []
    yline = []
    #y = np.random.random()
    #plt.scatter(i, y)
    #zdata = 15 * np.random.random(i)
    #xdata = np.sin(zdata) + 0.1 * np.random.randn(i)
    #ydata = np.cos(zdata) + 0.1 * np.random.randn(i)
    #ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    radius = 1
    u = np.linspace(0, 2 * np.pi, radius)
    v = np.linspace(0, np.pi, radius)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_surface(x, y, z, color='b')
    rrt = RRT.RRT(start=[0, 0], goal=[5, 10],
              randArea=[-20, 15], obstacleList=obstacleList)
    path = rrt.Planning(animation=False)
    print(path)
    for (x,y) in path:
        xline.append(x)
        yline.append(y)
    #print(path)
    zline = np.ones(len(path))
    #print(zline)
    #xline = np.sin(zline)
    #yline = np.cos(zline)
    print(len(xline))
    print(len(yline))
    print('\n\n\n\n\n')
    ax.plot3D(xline, yline, zline, 'gray')
    plt.pause(0.05)

plt.show()







