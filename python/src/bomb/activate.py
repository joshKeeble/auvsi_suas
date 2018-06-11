#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
PAYLOAD DEPLOYMENT ACTIVATION
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
import sys
import os


class RPiGPIO(object):

	def __init__(self,GPIO_ID):
		self.GPIO_ID = GPIO_ID

	#--------------------------------------------------------------------------


	def init_gpio(self):
		"""Initialize GPIO configuration"""
		pass

	#--------------------------------------------------------------------------

	def activate(self):
		pass