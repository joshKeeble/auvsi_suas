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



