#!/bin/env/python3
#-*- encoding: utf-8 -*-
import sys

paths = ['C:\Users\Rocket\Desktop\\','','C:\\Users\\Rocket\\Anaconda\\envs\\auvsi\\python36.zip',
'C:\\Users\\Rocket\\Anaconda\\envs\\auvsi\\DLLs', 'C:\\Users\\Rocket\\Anaconda\\envs\\auvsi\\lib',
'C:\\Users\\Rocket\\Anaconda\\envs\\auvsi', 'C:\\Users\\Rocket\\Anaconda\\envs\\auvsi\\lib\\site-packages']

for n in paths:
	sys.path.append(path)

import numpy as np
import pickle
import zlib

#from auvsi_suas.python.src.communications.osc_server import osc_server

try:
    import clr
    clr.AddReference("MissionPlanner")
    import MissionPlanner
    clr.AddReference("MAVLink")
    import MAVLink
except:
    print("ERROR: Not run within Mission Planner",file=sys.stderr)

print(1)

#import auvsi_suas.python.src.communications.osc_server as osc_server
#import auvsi_suas.python.src.communications.osc_client as osc_client
#import auvsi_suas.config as config

clr.AddReference("MissionPlanner.Utilities")
#Script.ChangeMode("Guided")



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
        MAV.setGuidedModeWP(wp)
    '''

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

    #--------------------------------------------------------------------------

    def fetch_battery_voltage(self):
        """Return the battery voltage"""
        return cs.battery_voltage

    #--------------------------------------------------------------------------

    def fetch_battery_remaining(self):
        """Return the remaining battery percentage"""
        return cs.battery_remaining

    #--------------------------------------------------------------------------

    def fetch_time_in_air(self):
        """Return the time in air of the uav"""
        return cs.timeInAir

    #--------------------------------------------------------------------------

    def fetch_mode(self):
        """Return the flying mode as string"""
        return cs.mode

    #--------------------------------------------------------------------------

    def is_armed(self):
        """Return whether the uav is armed"""
        return cs.armed

    '''


print(2)


def gs2mp_server_handle(channel,data):
    """TCP server data packet handler"""
    data = zlib.decompress(data)
    data = pickle.loads(data)
    data = np.asarray(data)
    print("Incoming server data:{}".format(data))
    if (channel == 'path'):
        for n in data:
            mp_interface.add_waypoint(n[0],n[1],n[2])

print(3)

def main():
	mp_interface = MPInterface()
	mp_interface.init_autonomous_mode()
	if not cs.mode == "Guided":
		Script.ChangeMode("Guided")

	gs2mp_server = osc_server.OSCServer('192.168.1.42',5005)
	gs2mp_server.init_server(gs2mp_server_handle)
	gs2mp_server.listen()