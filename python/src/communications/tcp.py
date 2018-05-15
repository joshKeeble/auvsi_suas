#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
TCP TEST
===============================================================================
Test the TCP socket connection
-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
import numpy as np
import threading
import pickle
import socket
import types
import time
import zlib
import sys
import os
import re

__author__ = "hal112358"

"""
===============================================================================
TCP MESSAGE OBJECT
===============================================================================
"""

class TCPMessage(object): ##################################################### FIX

	def __new__(self,data):
		data = self.process_type(data)
		return data

	#--------------------------------------------------------------------------

	def __name__(self):
		return 'TCPMessage'

	#--------------------------------------------------------------------------

	def process_type(self,data):
		data_dict = {
			type(str).__name__			:	self.cnvrt_str,
			type(int).__name__			:	self.cnvrt_int,
			type(float).__name__		:	self.cnvrt_float,
			type(np.ndarray).__name__	:	self.cnvrt_str,
		}
		return data_dict[type(data).__name__]()

	#--------------------------------------------------------------------------

	def compress(self,data):
		return zlib.compress(pickle.dumps(data,pickle.HIGHEST_PROTOCOL),9)

	#--------------------------------------------------------------------------

	def cnvrt_str(self,str_data):
		return b'\x00'+self.compress(str_data)

	#--------------------------------------------------------------------------

	def cnvrt_int(self,int_data):
		return b'\x01'+bytes([int_data])


"""
===============================================================================
CLIENT TEST
===============================================================================
"""
class TCPClient(object):
	def __init__(self,host,port):
		if not (re.match(r'^((\d){1,3}.){3}(\d{1,3})$',host,re.M|re.I)):
			raise Exception("Invalid host argument:{}".format(host))

		if not isinstance(port,int):
			raise Exception("Invalid port type:{}".format(
				type(port).__name__))

		if not (port >= 0 and port < 65535):
			raise Exception("Invalid port range (0-65535): {}".format(port))

		self.host      = host
		self.port      = port
		self.buffer    = 1024
		self.timeout   = 10
		self.connected = False
		self.init_socket()

	#--------------------------------------------------------------------------

	def init_socket(self):
		self.client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		self.client_socket.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
		self.client_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
		self.client_socket.settimeout(self.timeout)

	#--------------------------------------------------------------------------

	def _connect(self):
		self.client_socket.connect((self.host,self.port))
		self.connected = True

	#--------------------------------------------------------------------------

	def str_compress(self,data):
		return zlib.compress(pickle.dumps(data,pickle.HIGHEST_PROTOCOL),9)

	#--------------------------------------------------------------------------

	def send_data(self,data,close=False):

		if isinstance(data,str):
			data = self.str_compress(data)

		if not (self.connected):
			start_time = time.time()
			while (time.time()-start_time<=self.timeout):
				print(self.host,self.port)
				try:
					self.client_socket.connect((self.host,self.port))
					self.connected = True
					break
				except Exception as e:
					print("Error connecting: {}".format(e),file=sys.stderr)
		try:
			self.client_socket.send(data)
			server_data = self.client_socket.recv(self.buffer)
		except SocketError:
			self.client_socket.close()
			self.connected = False
		if close:
			self.client_socket.close()

"""
===============================================================================
SERVER TEST
===============================================================================
"""
class TCPServer(object):

	def __init__(self,host,port):
		if not (re.match(r'^((\d){1,3}.){3}(\d{1,3})$',host,re.M|re.I)):
			raise Exception("Invalid host argument:{}".format(host))

		if not isinstance(port,int):
			raise Exception("Invalid port type:{}".format(
				type(port).__name__))

		if not (port >= 0 and port < 65535):
			raise Exception("Invalid port range (0-65535): {}".format(port))

		self.host      = host
		self.port      = port
		self.buffer    = 1024
		self.timeout   = 10
		self.connected = False
		self.server_thread = None

	#--------------------------------------------------------------------------

	def init_socket(self):
		self.server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		self.server_socket.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
		self.server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
		self.server_socket.settimeout(self.timeout)
		self.server_socket.bind((self.host,self.port))
		print("Server bound at address ({}:{})".format(self.host,self.port),
			file=sys.stderr)

	#--------------------------------------------------------------------------

	def listen(self,handle,handle_args):
		self.init_socket()
		while True:
			self.server_socket.listen(1)
			connection,addr = self.server_socket.accept()

			while True:
				data = connection.recv(self.buffer)
				client_message = handle(data,handle_args)
				if not data:
					break
				if not isinstance(client_message,bytes):
					print("Non-byte client message type:{}".format(
						type(client_message).__name__))
					connection.send(b'\x00')
				else:
					connection.send(client_message)
			connection.close()

	#--------------------------------------------------------------------------

	def create_thread(self,handle,handle_args):
		self.server_thread = threading.Thread(target=self.listen,
			args=(handle,handle_args))
		self.server_thread.daemon = True

	#--------------------------------------------------------------------------

	def activate_thread(self,handle,handle_args):
		if not self.server_thread():
			self.create_thread(handle,handle_args)
		self.server_thread.start()
