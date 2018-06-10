#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
print(sys.version)

paths =  ['C:\Users\Rocket\Desktop\\','', 'C:\\Users\\Rocket\\Anaconda\\envs\\mission_planner\\python27.zip', 
'C:\\Users\\Rocket\\Anaconda\\envs\\mission_planner\\DLLs', 
'C:\\Users\\Rocket\\Anaconda\\envs\\mission_planner\\lib', 
'C:\\Users\\Rocket\\Anaconda\\envs\\mission_planner\\lib\\plat-win', 
'C:\\Users\\Rocket\\Anaconda\\envs\\mission_planner\\lib\\lib-tk', 
'C:\\Users\\Rocket\\Anaconda\\envs\\mission_planner', 
'C:\\Users\\Rocket\\Anaconda\\envs\\mission_planner\\lib\\site-packages']

print(sys.path)

for n in paths:
	sys.path.append(n)

print(sys.path)

import threading
import pickle
import socket
import struct
import time
import os


"""
===============================================================================
Client Object
===============================================================================
"""

class VideoStreamClient(object):

    def __init__(self,host,port):
        self.host = host
        self.port = port
        self.setup_client()

    #--------------------------------------------------------------------------

    def setup_client(self):
        self.client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.client_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        self.client_socket.connect((self.host,self.port))

    #--------------------------------------------------------------------------

    def send_frame(self,frame):
        frame_data = pickle.dumps(frame)
        frame_data = struct.pack("L",len(frame_data))+frame_data

        self.client_socket.sendall(frame_data)

        self.client_socket.recv(10)

"""
===============================================================================
Server Object
===============================================================================
"""

class VideoStreamServer(object):

    def __init__(self,host,port):
        self.host = host
        self.port = port
        self.setup_server()

    #--------------------------------------------------------------------------

    def setup_server(self):
        self.server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        self.server_socket.bind((self.host,self.port))
        print("Server activated")

    #--------------------------------------------------------------------------

    def listen(self,handle,handle_args):
        self.server_socket.listen(10)
        conn,addr = self.server_socket.accept()
        
        package_size = struct.calcsize("L")

        n_packets = 0

        while True:

            start_time = time.time()
            data = bytearray(b'')

            while (len(data)<package_size):
                data.extend(conn.recv(config.VIDEO_STREAM_RECV_SIZE))

            print(data)

            packed_msg_size = data[:package_size]

            data = data[package_size:]

            msg_size = struct.unpack("L",packed_msg_size)[0]

            while (len(data) < msg_size):
                data.extend(conn.recv(config.VIDEO_STREAM_RECV_SIZE))


            if len(data):

                packet = pickle.loads(data)

                client_message = handle(packet,handle_args)
                
                n_packets += 1
                print("Recieved packets: {}".format(n_packets))

                conn.send(client_message)

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
        MAV.setGuidedModeWP(wp)

    #--------------------------------------------------------------------------

    def fetch_mode(self):
        """Return the flying mode as string"""
        return cs.mode


print(2)


def gs2mp_server_handle(data,handle_args):
    """TCP server data packet handler"""

    print("Incoming server data:{}".format(data))
    #for n in data:
    #    mp_interface.add_waypoint(n[0],n[1],n[2])

print(3)

def main():
    mp_interface = MPInterface()
    mp_interface.init_autonomous_mode()
    if not cs.mode == "Guided":
        Script.ChangeMode("Guided")
    gs2mp_server = VideoStreamServer('192.168.1.69',5005)
    gs2mp_server.listen(gs2mp_server_handle,None)
    #gs2mp_server = osc_server.OSCServer('192.168.1.42',5005)
    #gs2mp_server.init_server(gs2mp_server_handle)
    #gs2mp_server.listen()




main()
