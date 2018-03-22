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
import sys
import os


__author__ = "hal112358"
   1 #!/usr/bin/env python
   2
   3 import socket
   4
   5
   6 TCP_IP = '127.0.0.1'
   7 TCP_PORT = 5005
   8 BUFFER_SIZE = 1024
   9 MESSAGE = "Hello, World!"
  10
  11 s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  12 s.connect((TCP_IP, TCP_PORT))
  13 s.send(MESSAGE)
  14 data = s.recv(BUFFER_SIZE)
  15 s.close()
  16
  17 print "received data:", data


"""
===============================================================================
CLIENT TEST
===============================================================================
"""
   1 #!/usr/bin/env python
   2
   3 import socket
   4
   5
   6 TCP_IP = '127.0.0.1'
   7 TCP_PORT = 5005
   8 BUFFER_SIZE = 20  # Normally 1024, but we want fast response
   9
  10 s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  11 s.bind((TCP_IP, TCP_PORT))
  12 s.listen(1)
  13
  14 conn, addr = s.accept()
  15 print 'Connection address:', addr
  16 while 1:
  17     data = conn.recv(BUFFER_SIZE)
  18     if not data: break
  19     print "received data:", data
  20     conn.send(data)  # echo
  21 conn.close()
