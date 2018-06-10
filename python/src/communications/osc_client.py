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
from pythonosc import osc_message_builder
from pythonosc import udp_client
import argparse
import random
import pickle
import json
import time
import zlib
import sys
import os
import re


"""
===============================================================================
OSC Client Object
===============================================================================
"""

class OSCClient(object):

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

    def init_client(self):
        self.client = udp_client.SimpleUDPClient(self.host,self.port)

    #--------------------------------------------------------------------------

    def send_data(self,data,channel="/filter"):
        self.client.send_message(channel,data)

#------------------------------------------------------------------------------

def main():
    import cv2
    video_data = cv2.VideoCapture(0)
    test_host = "127.0.0.1"
    test_port = 5005
    client    = OSCClient(test_host,test_port)
    client.init_client()
    while True:
        _,frame = video_data.read()
        frame = cv2.resize(frame,(50,50))
        frame = pickle.dumps(frame)
        frame = zlib.compress(frame)
        #print(frame)
        # print(frame)
        # frame = json.dumps(frame)
        # print(frame)
        client.send_data(frame)
        cv2.waitKey(1)

if __name__ == "__main__":
  main()