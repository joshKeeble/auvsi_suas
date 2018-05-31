#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
AUVSI SUAS User Interface
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
import MissionPlanner
import time
import clr
import sys
import os

clr.AddReference("MissionPlanner.Utilities")
Script.ChangeMode("Guided")

class MPInterface(object):

	def __init__(self):
		pass

	#--------------------------------------------------------------------------

	def init_autonomous_mode(self):
		Script.ChangeMode("Guided")

	#--------------------------------------------------------------------------

	def add_waypoint(self,lat,lng,alt):
		wp = MissionPlanner.Utilities.Locationwp()
		MissionPlanner.Utilities.Locationwp.lat.SetValue(wp,lat)
		MissionPlanner.Utilities.Locationwp.lat.SetValue(wp,lng)
		MissionPlanner.Utilities.Locationwp.lat.SetValue(wp,alt)
		MAV.setGuidedModeWP(wp)

	#--------------------------------------------------------------------------

	
