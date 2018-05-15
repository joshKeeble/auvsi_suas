#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
INTEROPERABILITY SYSTEM FUNCTIONS
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
import requests
import unittest
import os

from . import client
from . import exceptions
from . import types

class InteropInterface(object):

	def __init__(self):
		pass

	#--------------------------------------------------------------------------

	def login(self,server,username,password):
		self.client = Client(server, username, password)