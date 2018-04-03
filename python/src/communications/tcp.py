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
from enum import IntEnum
import numpy as np
import threading
import pickle
import socket
import types
import zlib
import sys
import os
import re
import time
import struct

__author__ = "hal112358"

"""
===============================================================================
Packets
===============================================================================
Establishes a protocol for the client and server to communicate by
-------------------------------------------------------------------------------
"""	

class PacketType(IntEnum):
	MESSAGE_SHORT = 0 # Use for short messages which do not benefit from compression
	MESSAGE_LONG  = 1 # Use for longer messages which will benefit from compression

class Packet():
	def __init__(self, id):
		self.id = id

	#--------------------------------------------------------------------------

	def serialize(self, tail=b""):
		return struct.pack("IB" + str(len(tail)) + "s", 1 + len(tail), self.id, tail)

	#--------------------------------------------------------------------------

	def deserialize(data):
		if len(data) < 4:
			return [data, None]
		length = struct.unpack_from("I", data)[0]
		if len(data) - struct.calcsize("I") < length:
			return [data, None]
		return [data[struct.calcsize("IB"):],
			Packet(struct.unpack_from("B", data, struct.calcsize("I"))[0])]

class MessageShortPacket(Packet):
	def __init__(self, message):
		self.message = message

		if len(message) >= 256:
			raise ValueError("Message is too long: {}".format(message))

		super().__init__(PacketType.MESSAGE_SHORT)

	def serialize(self):
		return super().serialize(struct.pack("B" + str(len(self.message)) + "s",
						     len(self.message),
						     self.message.encode()))

	#--------------------------------------------------------------------------

	def deserialize(data):
		length = struct.unpack_from("B", data)[0]
		return [data[struct.calcsize("B") + length:],
			MessageShortPacket(struct.unpack_from(str(length) + "s",
							      data,
							      struct.calcsize("B"))[0].decode())]

class MessageLongPacket(Packet):
	def __init__(self, message):
		self.message = message
		super().__init__(PacketType.MESSAGE_LONG)

	#--------------------------------------------------------------------------

	def serialize(self):
		compressed = zlib.compress(self.message.encode(), 9)
		return super().serialize(struct.pack("I" + str(len(compressed)) + "s",
						     len(compressed),
						     compressed))

	#--------------------------------------------------------------------------

	def deserialize(data):
		length = struct.unpack_from("I", data)[0]
		compressed = struct.unpack_from(str(length) + "s", data, struct.calcsize("I"))[0]
		return [data[struct.calcsize("I") + length:],
			MessageLongPacket(zlib.decompress(compressed).decode())]

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

	def send_packet(self,packet,close=True):
		#self.init_socket()
		if not self.connected:
			self._connect()
		self.client_socket.send(packet.serialize())
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
			data = b"" # Holds unparsed bytes from client
			while True:
				data += connection.recv(self.buffer)
				if not data:
					break
				while len(data) > 0:
					data, packet = Packet.deserialize(data)
					if packet == None:
						break
					if packet.id == PacketType.MESSAGE_SHORT:
						data, packet = MessageShortPacket.deserialize(data)
						print("Recieved: ", packet.message)
					elif packet.id == PacketType.MESSAGE_LONG:
						data, packet = MessageLongPacket.deserialize(data)
						print("Recieved: ", packet.message)
				#connection.send(data)
			#connection.close()

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
	time.sleep(1) # Give server time to startup

	c = TCPClient('127.0.0.1',5005)
	for i in range(1000):
		c.send_packet(MessageShortPacket("{}".format(i)), False)

	c.send_packet(MessageLongPacket("ABCD" * 16), True)

	time.sleep(1) # Give server time to finish with our client

if __name__ == "__main__":
	main()
