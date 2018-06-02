#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
AUVSI SUAS Mission Planner Interface
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
import numpy as np 
import pickle
import time
import zlib
import sys
import os

try:
    import MissionPlanner
    import clr
except:
    pass


import auvsi_suas.python.src.communications.osc_server as osc_server
import auvsi_suas.python.src.communications.osc_client as osc_client
import auvsi_suas.config as config

#clr.AddReference("MissionPlanner.Utilities")
#Script.ChangeMode("Guided")

def gs2mp_server_handle(channel,data):
    """TCP server data packet handler"""
    data = zlib.decompress(data)
    data = pickle.loads(data)
    data = np.asarray(data)
    print("Incoming server data:{}".format(data))
    

class MPInterface(object):

    def __init__(self):
        self.vehicle_lat = 38.147404206618816
        self.vehcile_lng =  -76.4277855321988
        self.ground_station_host = config.GROUND_STATION_HOST
        self.gs2mp_port = config.GROUND_STATION2MISSION_PLANNER_PORT
        self.mission_planner_host = config.MISSION_PLANNER_HOST 
        self.mp2gs_port = config.MISSION_PLANNER2GROUND_STATION_PORT

    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------

    def init_gs2mp_server(self):
        self.gs2mp_server = osc_server.OSCServer(self.ground_station_host,
            self.gs2mp_port)
        print("Mission Planner server connected at {}:{}".format(
            self.ground_station_host,self.gs2mp_port))
        self.gs2mp_server.init_server(gs2mp_server_handle)
        self.gs2mp_server.activate_listen_thread()
        self.ms2gs_client = osc_client.OSCClient(
            self.mission_planner_host,self.mp2gs_port)
        self.ms2gs_client.init_client()
        while True:
            ################################################################### Create update function
            telemetry = [self.vehicle_lat,self.vehcile_lng]
            compressed_data = zlib.compress(pickle.dumps(
                np.asarray(telemetry)))
            self.ms2gs_client.send_data(compressed_data)

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

    
