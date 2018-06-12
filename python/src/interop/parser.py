#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
AUVSI SUAS Interop Parser
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
import numpy as np 
import sys
import os

import auvsi_suas.python.src.communications.osc_client as osc_client
import auvsi_suas.python.src.communications.osc_server as osc_server
import auvsi_suas.python.src.communications.test_link as tcp_test
import auvsi_suas.python.src.stealth.stealth_mode as stealth
import auvsi_suas.python.src.interop.client as client
import auvsi_suas.config as config

"""
===============================================================================

===============================================================================
"""

class InteropParser(object):

    def __init__(self,client):
        self.client = client

        self.stealth_mode = stealth.StealthMode()
        self.stealth_mode.set_min_altitude(150)
        self.stealth_mode.set_max_altitude(400)

    #--------------------------------------------------------------------------

    def run_stealth_mode(self):
        global current_position
        i = 0
        gs2mp_client = osc_client.OSCClient(config.GROUND_STATION_HOST,
            config.GROUND_STATION2MISSION_PLANNER_PORT)
        gs2mp_client.init_client()
        print("Ground Station to Mission Planner Client activated",
            file=sys.stderr)
        while True:
            if self.stealth_mode_activated:
                obstacles = []
                for n in self.stealth_mode.obstacles:
                    try:
                        obstacles.append((n[0],n[1],n[2]))
                    except:
                        obstacles.append((n[0],n[1],n[2]))
                # print(obstacles)
                optimal_path = self.stealth_mode.find_path(obstacles,
                    self.stealth_mode.mission_waypoints,
                    self.stealth_mode.current_position,
                    self.stealth_mode.boundaries)
                try:
                    compressed_data = zlib.compress(pickle.dumps(np.asarray(optimal_path)))
                    gs2mp_client.send_data(compressed_data,channel='path')
                    # print("Data sent:{}".format(compressed_data))

                except Exception as e:
                    exc_type,exc_obj,exc_tb = sys.exc_info()
                    fname = os.path.split(
                        exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type,fname,exc_tb.tb_lineno,e,
                        file=sys.stderr)
                i += 1
            else:
                time.sleep(1e-2)

    #--------------------------------------------------------------------------

    def fetch_mission(self):
        """Main function for processing interop data"""
        #self.init_telemetry_server()
        if config.INTEROP_USE_ASYNC:
            mission = self.interop_client.get_missions().result()[0]
        else:
            mission = self.interop_client.get_missions()[0]

        self.stealth_mode.set_home_location(mission.home_pos)
        self.stealth_mode.update_waypoints(mission.mission_waypoints)
        self.stealth_mode.set_boundaries(mission.fly_zones)

        # Create initial obstacles
        if config.INTEROP_USE_ASYNC:
            async_future = self.interop_client.get_obstacles()
            async_stationary, async_moving = async_future.result()
            self.all_obstacles = async_stationary+async_moving
        else:
            obstacle_list = self.interop_client.get_obstacles()
            stationary,moving = obstacle_list
            self.all_obstacles = stationary+moving
        self.stealth_mode.update_obstacles(self.all_obstacles)

        # Set the current position of the vehicle
        self.stealth_mode.set_current_position(current_position)


