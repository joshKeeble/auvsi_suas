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
import zlib
import sys
import os
import re

__author__ = "hal112358"



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

		#self.init_socket()

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

	def send_data(self,data,close=True):
		self.init_socket()
		if isinstance(data,str):
			data = self.str_compress(data)
			print(data)
		if not self.connected:
			print(self.host,self.port)
			self.client_socket.connect((self.host,self.port))

		self.client_socket.send(data)
		#server_data = self.client_socket.recv(self.buffer)
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

	#--------------------------------------------------------------------------

	def init_socket(self):
		self.server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		self.server_socket.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
		self.server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
		self.server_socket.settimeout(self.timeout)
		self.server_socket.bind((self.host,self.port))
		print("server bound at address ({}:{})".format(self.host,self.port),
			file=sys.stderr)

	#--------------------------------------------------------------------------

	def listen(self,handle,handle_args):
		self.init_socket()
		while True:
			self.server_socket.listen(1)
			connection,addr = self.server_socket.accept()

			while True:
				data = connection.recv(self.buffer)
				if not data:
					break
				#connection.send(data)
			connection.close()

def test_handle(x,handle_args):
	print(x)

def activate_server():
	host = "127.0.0.1"
	port = 5005

	server = TCPServer(host,port)
	server.listen(test_handle,42)

def activate_server_thread():
	server_thread = threading.Thread(target=activate_server,args=())
	server_thread.daemon = True
	server_thread.start()

def main():
	activate_server_thread()

	c = TCPClient('127.0.0.1',5005)
	for i in range(1000):
		c.send_data("{}".format(i))

if __name__ == "__main__":
	main()
