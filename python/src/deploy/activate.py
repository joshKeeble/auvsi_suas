#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
PAYLOAD DEPLOYMENT ACTIVATION
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
import sys
import os


class PayloadDeployment(object):

	def __init__(self):
		self.drop_zone_lat = 0
		self.drop_zone_lng = 0
		self.drop_height = 400

	#--------------------------------------------------------------------------

	def update_drop_zone(self,lat,lng):
		"""Update the drop zone location"""
		print("Drop zone",lat,lng)
		self.drop_zone_lat = lat
		self.drop_zone_lng = lng

	#--------------------------------------------------------------------------

	def send_data(self,client):
		"""Send the information to mission planner"""
		client.send_data(("drop",(self.drop_zone_lat,
			self.drop_zone_lng,self.drop_height)))

