#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
GIMBAL FUNCTIONS
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
import numpy as np
import sys
import os

try:
	import auvsi_suas.python.src.communications.osc_client as osc_client
	import auvsi_suas.python.src.communications.osc_server as osc_server
	import auvsi_suas.python.src.communications.test_link as tcp_test
    import auvsi_suas.config as config
    
except ModuleNotFoundError as e:
    cwd = os.getcwd().split(os.path.sep)
    project_dir = os.path.sep.join(cwd[:cwd.index("auvsi_suas")])
    print('{}\n\nRun "export PYTHONPATH=$PYTHONPATH:{}"'.format(e,
                project_dir),file=sys.stderr)

"""
===============================================================================
Gimbal Object Functions
===============================================================================
"""

class Gimbal(object):

	def __init__(self):

		self.gimbal_host 				= config.GIMBAL_HOST 
		self.gimbal2ground_station_port = config.GIMBAL2GROUND_STATION_PORT
		self.ground_station_host 		= config.GROUND_STATION_HOST
		self.ground_station2gimbal_port = config.GROUND_STATION2GIMBAL_PORT
		self.connection_tries			= 10

	#--------------------------------------------------------------------------

	def init_gimbal(self):
		

	#--------------------------------------------------------------------------

	def init_gimbal2ground_station_client(self):
		pass

	#--------------------------------------------------------------------------

	def init_ground_station2gimbal_server(self):
		connection_tries = 0
		while not tcp_test(self.ground_station_host,self.ground_station2gimbal_port):
			time.sleep(0.01)
			print("Connection to {}:{} unsucessful, attempt:{}".format(
				self.ground_station_host,self.ground_station2gimbal_port,
				connection_tries),file=sys.stderr)
			if (self.connection_tries > self.n_connection_attemps):
				connected = False
				break
		if connected:
			self.ground_station2gimbal_server = osc_server.OSCServer(
				self.ground_station_host,self.ground_station2gimbal_port)
			self.ground_station2gimbal_server.init_server(self.gimbal_handle)
			self.ground_station2gimbal_server.activate_listen_thread()
		else:
			print("Gimbal could not connect to ground station",file=sys.stderr)

	#--------------------------------------------------------------------------

	def gimbal_handle(channel,data):
		(uav_latitude,uav_longitude) = data
