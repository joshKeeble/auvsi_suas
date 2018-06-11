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
import threading
import pickle
import zlib
import time
import sys
import os

import auvsi_suas.python.src.communications.tcp as TCP

__author__ = "hal112358"

TEST_HOST = "127.0.0.1"
TEST_PORT = 5005

"""
-------------------------------------------------------------------------------
SERVER TEST
-------------------------------------------------------------------------------
"""

def test_handle(data,handle_args):
    try:
        print(data)
        data = pickle.loads(zlib.decompress(data))
        print(data,file=sys.stderr)
    except Exception as e:
        print(e)
    return b'\x01'

#------------------------------------------------------------------------------

def server_test():
    server = TCP.TCPServer(TEST_HOST,TEST_PORT)
    server.listen(test_handle,42)

#------------------------------------------------------------------------------

def activate_server_thread():
    server_thread = threading.Thread(target=server_test,args=())
    server_thread.daemon = True
    server_thread.start()

"""
-------------------------------------------------------------------------------
CLIENT TEST
-------------------------------------------------------------------------------
"""

def client_test():
    client = TCP.TCPClient(TEST_HOST,TEST_PORT)
    while True:
        client.send_data(str(time.time()))

"""
-------------------------------------------------------------------------------
TCP MAIN
-------------------------------------------------------------------------------
"""

def main():
    activate_server_thread()
    time.sleep(0.5)
    client_test()

if __name__=="__main__":
    main()
