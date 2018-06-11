#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
FETCH GPS
===============================================================================
Given current telemetry, estimate GPS of target
-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
import numpy as np
import cv2
import sys
import os

__author__ = "gndctrl2mjrtm"

import auvsi_suas.python.src.interfaces.gpsd as gpsd

"""
===============================================================================

===============================================================================
"""

class GPSEstimator(self):

	def __init__(self):
		self.vehicle_gps = [0,0]

	def check_gps(self):
		"""Check if the GPS is connected and operating"""
		pass

	def fetch_gps(self):
		pass
