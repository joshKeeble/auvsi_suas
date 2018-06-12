#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
print(sys.version)

paths =  ['C:\\Users\\soffer\\Desktop','C:\\Users\\soffer\\AppData\\Local\\Continuum\\anaconda3\\envs\\mission_planner\\python27.zip', 
'C:\\Users\\soffer\\AppData\\Local\\Continuum\\anaconda3\\envs\\mission_planner\\DLLs', 
'C:\\Users\\soffer\\AppData\\Local\\Continuum\\anaconda3\\envs\\mission_planner\\lib', 
'C:\\Users\\soffer\\AppData\\Local\\Continuum\\anaconda3\\envs\\mission_planner\\lib\\plat-win', 
'C:\\Users\\soffer\\AppData\\Local\\Continuum\\anaconda3\\envs\\mission_planner\\lib\\lib-tk', 
'C:\\Users\\soffer\\AppData\\Local\\Continuum\\anaconda3\\envs\\mission_planner', 
'C:\\Users\\soffer\\AppData\\Local\\Continuum\\anaconda3\\envs\\mission_planner\\lib\\site-packages']

for n in paths:
	sys.path.append(n)

import threading
import pickle
import socket
import struct
import time
import os


import auvsi_suas.python.src.communications.mp_communications as mp_coms
try:
    import clr
    clr.AddReference("MissionPlanner")
    import MissionPlanner
    clr.AddReference("MAVLink")
    import MAVLink
except:
    print("ERROR: Not run within Mission Planner")


#import auvsi_suas.config as config

clr.AddReference("MissionPlanner.Utilities")


print(2)


class MPInterface(object):

    def __init__(self):
        pass

    #--------------------------------------------------------------------------

    def signal_servo(self,servo=None,pwm=None):
        """
        The servo is servo 9, pwm is 1200 for open/drop, pwm is 1500 for closed.
        """
        MAV.doCommand(MAVLink.MAV_CMD.DO_SET_SERVO,servo,pwm,0,0,0,0,0)

    #--------------------------------------------------------------------------

    def open_payload(self):
        """Open the payload claw"""
        self.signal_servo(servo=9,pwm=1200)

    #--------------------------------------------------------------------------

    def close_payload(self):
        """Close the payload claw"""
        self.signal_servo(servo=9,pwm=1500)

    #--------------------------------------------------------------------------

    def init_autonomous_mode(self):
        if not (self.fetch_mode() == "Guided"):
            Script.ChangeMode("Guided")

    #--------------------------------------------------------------------------

    def init_gs2mp_server(self):
        """Initialize ground station to mission planner server"""
        self.gs2mp_server = osc_server.OSCServer(config.GROUND_STATION_HOST,
            config.GROUND_STATION2MISSION_PLANNER_PORT)
        self.gs2mp_server.init_server(gs2mp_server_handle)
        self.gs2mp_server.activate_listen_thread()

    #--------------------------------------------------------------------------

    def add_waypoint(self,lat,lng,alt):
        """Add a waypoint"""
        wp = MissionPlanner.Utilities.Locationwp()
        MissionPlanner.Utilities.Locationwp.lat.SetValue(wp,lat)
        MissionPlanner.Utilities.Locationwp.lng.SetValue(wp,lng)
        MissionPlanner.Utilities.Locationwp.alt.SetValue(wp,alt)
        MAV.setAutoModeWP(wp)

    #--------------------------------------------------------------------------

    def fetch_mode(self):
        """Return the flying mode as string"""
        return cs.mode




def gs2mp_server_handle(data,handle_args):

    print("Incoming server data:{}".format(data))
    return b'1'

host = 'localhost'
port = 4012


def main():
    mp_interface = MPInterface()
    mp_interface.init_autonomous_mode()
    if not cs.mode == "Guided":
        Script.ChangeMode("Auto")
    print("Ready to recieve gps coordinates")
    gs2mp_server = mp_coms.MissionPlannerServer(host,port)
    gs2mp_server.listen(gs2mp_server_handle,0x00)
    #gs2mp_server = osc_server.OSCServer('192.168.1.42',5005)
    #gs2mp_server.init_server(gs2mp_server_handle)
    #gs2mp_server.listen()




main()
