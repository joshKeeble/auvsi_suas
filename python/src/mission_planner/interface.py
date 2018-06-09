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

    def fetch_roll(self):
        """Return the roll of the uav (deg)"""
        return cs.roll

    #--------------------------------------------------------------------------

    def fetch_pitch(self):
        """Return the pitch of the uav (deg)"""
        return cs.pitch

    #--------------------------------------------------------------------------

    def fetch_yaw(self):
        """Return the pitch of the uav (deg)"""
        return cs.yaw

    #--------------------------------------------------------------------------

    def fetch_latitude(self):
        """Return the latitude of the uav (deg)"""
        return cs.lat

    #--------------------------------------------------------------------------

    def fetch_longitude(self):
        """Return longitude of the uav (deg)"""
        return cs.lng

    #--------------------------------------------------------------------------

    def fetch_gps(self):
        """Return GPS of the uav (deg)"""
        if not self.fetch_gps_status():
            raise SystemError("Warning, GPS not activated")
        return [cs.lat,cs.lng]

    #--------------------------------------------------------------------------

    def fetch_gps_status(self):
        """Return the status of the gps"""
        return cs.gpsstatus

    #--------------------------------------------------------------------------

    def fetch_alt(self):
        """Return the altitude of the uav"""
        return cs.alt

    #--------------------------------------------------------------------------

    def fetch_airspeed(self):
        """Return the airspeed of the uav"""
        return cs.airspeed

    #--------------------------------------------------------------------------

    def fetch_ground_speed(self):
        """Return the groundspeed of the uav"""
        return cs.groundspeed

    #--------------------------------------------------------------------------

    def fetch_vertical_speed(self):
        """Return the airpseed of the uav"""
        return cs.verticalspeed

    #--------------------------------------------------------------------------

    def fetch_wind_direction(self):
        """Return the direction of the wind (deg)"""
        return cs.wind_dir

    #--------------------------------------------------------------------------

    def fetch_wind_velocity(self):
        """Return the velocity of the wind"""
        return cs.wind_vel

    #--------------------------------------------------------------------------

    def fetch_acceleration(self):
        """Return the acceleration fo the uav [ax,ay,az]"""
        return [cs.ax,cs.ay,cs.az]

    #--------------------------------------------------------------------------

    def fetch_gyroscope(self):
        """Return the gyroscope readings of the uav [gx,gy,gz]"""
        return [cs.gx,cs.gy,cs.gz]

    #--------------------------------------------------------------------------

    def fetch_magnetometer(self):
        """Return the magnetometer readings of the uav [mx,my,mz]"""
        return [cs.mx,cs.my,cs.mz]

    #--------------------------------------------------------------------------

    def fetch_input_channel1(self):
        """Return the input channel 1"""
        return cs.chx1in

    #--------------------------------------------------------------------------

    def fetch_input_channel2(self):
        """Return the input channel 2"""
        return cs.chx2in

    #--------------------------------------------------------------------------

    def fetch_input_channel3(self):
        """Return the input channel 3"""
        return cs.chx3in

    #--------------------------------------------------------------------------

    def fetch_input_channel4(self):
        """Return the input channel 4"""
        return cs.chx4in

    #--------------------------------------------------------------------------

    def fetch_input_channel5(self):
        """Return the input channel 5"""
        return cs.chx5in

    #--------------------------------------------------------------------------

    def fetch_input_channel6(self):
        """Return the input channel 6"""
        return cs.chx6in

    #--------------------------------------------------------------------------

    def fetch_input_channel7(self):
        """Return the input channel 7"""
        return cs.chx7in

    #--------------------------------------------------------------------------

    def fetch_input_channel8(self):
        """Return the input channel 8"""
        return cs.chx8in

    #--------------------------------------------------------------------------

    def fetch_output_channel1(self):
        """Return the output channel 1"""
        return cs.chx1out

    #--------------------------------------------------------------------------

    def fetch_output_channel2(self):
        """Return the output channel 1"""
        return cs.chx2out




rxrssi  float    
chx1in, chx2in, .... chx8in     float   Input Channels from 1 to 8
ch1out, chx2out, .... chx8out   float   Output Channel form 1 to 8
nav_roll    float   Roll Target (deg)
nav_pitch   float   Pitch Target (deg)
nav_bearing     float   Bearing target (deg)
target_bearing  float   Bearing Target (deg)
wp_dist     float   Distance to Next Waypoint (dist)
alt_error   float   Altitude Error (dist)
ber_error   float   Bearing Error (dist)
aspd_error  float   Airspeed Error (speed)
wpno    float   Flying Mode
mode    String  Flying Mode
dimbrate    float   Climb Rate (speed)
tot     int     Time over target (sec)
distTraveled    float   Distance Traveled (dist)
timeInAir   float   Time in Air (sec)
turnrate    float   Turn Rate (speed)
radius  float   Turn Radius (dist)
battery_voltage     float   Battery Voltage (volt)
battery_remaining   float   Battery Remaining (%)
current     float   battery Current (Amps)
HomeAlt     float    
DistToHome  float   Absolute Pressure Value
press_abs   float   Absolute Pressure Value
sonarrange  float   Sonar Range (meters)
sonarVoltage    float   Sonar Voltage (volt)
armed

    
