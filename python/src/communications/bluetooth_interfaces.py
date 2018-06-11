#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
OSC Client
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
import bluetooth
import sys
import os
# Uses Bluez for Linux
#
# sudo apt-get install bluez python-bluez
# 
# Taken from: https://people.csail.mit.edu/albert/bluez-intro/x232.html
# Taken from: https://people.csail.mit.edu/albert/bluez-intro/c212.html

class BTClient(object):

  def __init__(self,host,port):
    self.host = host
    self.port = port

  def lookUpNearbyBluetoothDevices():
    nearby_devices = bluetooth.discover_devices()
    for bdaddr in nearby_devices:
      print(str(bluetooth.lookup_name( bdaddr )) + " [" + str(bdaddr) + "]")

  def send_data(self,data):
    bt_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    bt_socket.connect((self.host,self.port))
    bt_socket.send(data)
    bt_socket.close()




class BTInterface(object):

  def __init__(self,host,port):
    self.host = host
    self.port = port

def receiveMessages():
  server_sock=bluetooth.BluetoothSocket(bluetooth.RFCOMM)
  
  server_sock.bind((self.host,self.port))
  server_sock.listen(1)
  
  client_sock,address = server_sock.accept()
  print("Accepted connection from {}".format(str(address)),
    file=sys.stderr)
  
  data = client_sock.recv(1024)
  print("received [%s]" % data)
  
  client_sock.close()
  server_sock.close()
  
def sendMessageTo(targetBluetoothMacAddress):
  port = 1
  sock=bluetooth.BluetoothSocket( bluetooth.RFCOMM )
  sock.connect((targetBluetoothMacAddress, port))
  sock.send("hello!!")
  sock.close()
  
def lookUpNearbyBluetoothDevices():
  nearby_devices = bluetooth.discover_devices()
  for bdaddr in nearby_devices:
    print(str(bluetooth.lookup_name( bdaddr )) + " [" + str(bdaddr) + "]")
    
    
lookUpNearbyBluetoothDevices()