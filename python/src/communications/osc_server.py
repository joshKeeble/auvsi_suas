#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
OSC Server
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
from pythonosc import dispatcher
from pythonosc import osc_server
from Crypto.Cipher import AES
import numpy as np
import threading
import pickle
import math
import zlib
import cv2
import sys
import os
import re




def handle(channel,data):
    data = zlib.decompress(data)
    data = pickle.loads(data)
    data = np.asarray(data,dtype=np.uint8)
    # print(data)
    cv2.imshow("data",data)
    cv2.waitKey(20)

"""
===============================================================================
OSC Server Object
===============================================================================
"""

class OSCServer(object):

    def __init__(self,host,port):
        if not (re.match(r'^((\d){1,3}.){3}(\d{1,3})$',host,re.M|re.I)):
            raise Exception("Invalid host argument:{}".format(host))
        if not isinstance(port,int):
            raise Exception("Invalid port type:{}".format(
                type(port).__name__))
        if not (port >= 0 and port < 65535):
            raise Exception("Invalid port range (0-65535): {}".format(port))
        self.host = host
        self.port = port

    #--------------------------------------------------------------------------

    def init_server(self,handle):
        self.handler = dispatcher.Dispatcher()
        self.handler.map("/filter", handle)
        self.server = osc_server.ThreadingOSCUDPServer(
            (self.host,self.port),self.handler)

    #--------------------------------------------------------------------------

    def print_volume_handler(self,unused_addr,args,volume):
        print("VH [{0}] ~ {1}".format(args[0], volume))

    #--------------------------------------------------------------------------

    def print_compute_handler(self,unused_addr,args,volume):
        try:
            print("CH [{0}] ~ {1}".format(args[0], args[1](volume)))
        except ValueError:
            pass

    #--------------------------------------------------------------------------

    def listen(self):
        self.server.serve_forever()

    #--------------------------------------------------------------------------

    def activate_listen_thread(self):
        self.listen_thread = threading.Thread(target=self.listen,args=())
        self.listen_thread.daemon = True
        self.listen_thread.start()

#------------------------------------------------------------------------------

def main():
    test_host = "127.0.0.1"
    test_port = 5005
    server    = OSCServer(test_host,test_port)
    server.init_server(handle)
    server.listen()

if __name__ == "__main__":
    main()