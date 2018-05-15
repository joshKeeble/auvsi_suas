#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
TEST CONNECTION LINK
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
import socket
import sys
import re
import os


def tcp_connection_test(host,port):
    """Test a connection to a host and port"""
    if not (re.match(r'^((\d){1,3}.){3}(\d{1,3})$',host,re.M|re.I)):
            raise Exception("Invalid host argument:{}".format(host))

    if not isinstance(port,int):
        raise Exception("Invalid port type:{}".format(
            type(port).__name__))

    if not (port >= 0 and port < 65535):
        raise Exception("Invalid port range (0-65535): {}".format(port))
        
    n_attempts     = 10
    socket_timeout = 1
    connection     = False

    test_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    test_socket.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    test_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    test_socket.settimeout(socket_timeout)

    for i in range(n_attempts):
        try:
            test_socket.bind((host,port)) 
            print("Sucessful Attempt:{} Connection to {}:{}".format(
                i,host,port),file=sys.stderr)
            connection = True
            break
        except Exception as e: 
            print("Failed Attmept:{} Connection to {}:{}".format(
                i,host,port),file=sys.stderr)
    test_socket.close()

    return connection

def test():
    tcp_connection_test("127.0.3.1",5005)

if __name__ == "__main__":
    test()