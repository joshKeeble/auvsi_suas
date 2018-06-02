#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
Simulation for UAV Mapping
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import math
import time
import sys
import os

import auvsi_suas.python.src.communications.osc_server as osc_server
import auvsi_suas.python.src.communications.osc_client as osc_client
import auvsi_suas.python.src.stealth.smoothed_RRT as smoothed_RRT

import auvsi_suas.python.src.communications.tcp as tcp
import auvsi_suas.python.src.stealth.rdp as rdp
import auvsi_suas.config as config


"""
===============================================================================
Stealth Mode to Avoid Obstacles Object Funcions
===============================================================================
"""

class StealthMode(object):

    def __init__(self):
        # Radius of the earth at location to estimate position
        self.earth_radius       = 20900000

        # Number of smooting iterations
        self.max_iterations     = 20000

        # Distance in ft from a waypoint to qualify as being met
        self.waypoint_threshold = 10

        # Scale by which to reduce the map
        self.path_scale         = 1

        self.current_position   = np.asarray([0,0]) ########################### FIX

        # The waypoints required to be met by the mission
        self.mission_waypoints  = []

        # The obstacles, both moving and stationary
        self.obstacles          = []

    #--------------------------------------------------------------------------

    def init_gs2mp_client(self):
        """Initialize mission planner TCP client"""
        print("{}:{}".format(config.GROUND_STATION_HOST,config.GROUND_STATION2MISSION_PLANNER_PORT))
        self.gs2mp_client = osc_client.OSCClient(config.GROUND_STATION_HOST,
            config.MISSION_PLANNER2GROUND_STATION_PORT)
        self.gs2mp_client.init_client()
        

    #--------------------------------------------------------------------------

    def init_mission_path(self,fly_zones,home_position,mission_waypoints):
        """Set the mission parameters"""
        self.fly_zones = fly_zones
        self.home_position = home_position
        self.mission_waypoints = mission_waypoints

    #--------------------------------------------------------------------------

    def prep_obstacles(self,obstacle):
        """Prepare obstacles for processing"""
        return self.latlng2ft(obstacle.latitude,obstacle.longitude)

    #--------------------------------------------------------------------------

    def latlng2ft(self,lat,lng):
        """Convert latitude and longitude to distance from ground station"""
        d_lat = self.angle2rad(self.home_lat-lat)
        d_lng = self.angle2rad(self.home_lng-lng)
        return np.array([self.earth_radius*np.sin(d_lat),
            self.earth_radius*np.sin(d_lng)])

    #--------------------------------------------------------------------------

    def ft2latlng(self,x,y):
        """Convert ft. coordifnates to longitude and latitude"""
        lat = self.home_lat-self.rad2angle(np.arcsin(x/self.earth_radius))
        lng = self.home_lng-self.rad2angle(np.arcsin(y/self.earth_radius))
        return (lat,lng)

    #--------------------------------------------------------------------------

    def update_obstacles(self,obstacle_list):
        temp_obstacles = []
        for n in obstacle_list:
            ox,oy = self.latlng2ft(n.latitude,n.longitude)
            try:
                radius = n.cylinder_radius
            except:
                radius = n.sphere_radius
            temp_obstacles.append([ox,oy,radius])
        self.obstacles = temp_obstacles
        del temp_obstacles

    #--------------------------------------------------------------------------

    def update_waypoints(self,waypoints):
        n_waypoints = len(waypoints)
        processed_waypoints = np.zeros((n_waypoints,2))
        for n in waypoints:
            processed_waypoints[n.order-1] = self.latlng2ft(n.latitude,
                n.longitude)
        self.mission_waypoints = processed_waypoints

    #--------------------------------------------------------------------------

    def set_home_location(self,home_waypoint):
        """Setup the location for ground station"""
        self.home_lat = home_waypoint.latitude
        self.home_lng = home_waypoint.longitude

    #--------------------------------------------------------------------------

    def set_current_position(self,current_position):
        """set the current position of the vehicle"""
        self.current_position = np.asarray(current_position)

    #--------------------------------------------------------------------------

    def eliminate_waypoints(self,waypoints):
        """Get rid of the waypoints if close enough"""
        distance = np.linalg.norm(waypoints[1]-waypoints[0])
        if (distance <= self.waypoint_threshold/self.path_scale):
            waypoints = waypoints[2:]
        waypoints  = np.insert(waypoints,0,self.current_position,axis=0)
        return waypoints

    #--------------------------------------------------------------------------

    def angle2rad(self,angle):
        """Convert angle to radian"""
        return (2*np.pi*angle)/360

    #--------------------------------------------------------------------------

    def rad2angle(self,rad):
        """Convert angle to radian"""
        return (360*rad)/(2*np.pi)

    #--------------------------------------------------------------------------

    def fetch_line(self,p1,p2,stride=2):
        """Create list of points from point 1 to point 2"""
        line = []
        (x1,y1) = p1
        (x2,y2) = p2
        slope = (y2-y1)/max((x2-x1),1e-3)
        for i in range(0,int(x2-x1),stride):
            line.append((x1+i,slope*(x1+i)+y1))
        return line

    #--------------------------------------------------------------------------

    def check_if_collision(self,p1,p2):
        """Check if there is an obstacle in the path"""
        line = self.fetch_line(p1,p2)
        collision = False
        for o in self.obstacles:
            for n in line:
                if (np.linalg.norm(np.asarray(n)-np.asarray(o[:2])) <= o[2]/self.path_scale):
                    collision = True
        return collision

    #--------------------------------------------------------------------------

    def check_waypoints_obstacles(self,mission_waypoints,obstacles):
        """Check for waypoints within obstacles"""
        updated_mission_waypoints = mission_waypoints.copy()
        for i,n in enumerate(updated_mission_waypoints):
            for j,o in enumerate(obstacles):
                if (np.equal(o[:2],n).all()):
                    if (i != len(updated_mission_waypoints)-1):
                        dx = updated_mission_waypoints[i+1][0]-o[0]
                        dy = updated_mission_waypoints[i+1][1]-o[1]
                        alpha = np.arctan(max(dy,1e-3)/max(dx,1e-3))
                        new_x = o[0]+o[2]*np.sin(alpha)
                        new_y = o[1]+o[2]*np.cos(alpha)
                        updated_mission_waypoints[i] = [new_x,new_y]
                    else:
                        dx = updated_mission_waypoints[i-1][0]-o[0]
                        dy = updated_mission_waypoints[i-1][1]-o[1]
                        alpha = np.arctan(dy/dx)
                        new_x = o[0]+o[2]*np.sin(alpha)
                        new_y = o[1]+o[2]*np.cos(alpha)
                        updated_mission_waypoints[i] = [new_x,new_y]
                elif (np.linalg.norm((n[1]-o[1],n[0]-o[0]))<=o[2]):
                    dx = n[0]-o[0]
                    dy = n[1]-o[1]
                    alpha = np.arctan(dy/max(dx,1e-2))
                    new_x = o[0]+o[2]*np.sin(alpha)
                    new_y = o[1]+o[2]*np.cos(alpha)
                    updated_mission_waypoints[i] = [new_x,new_y]
        return np.asarray(updated_mission_waypoints)

    #--------------------------------------------------------------------------

    def find_path(self,obstacles,mission_waypoints,current_position):
        """Find the optimal path"""
        if not isinstance(current_position,np.ndarray):
            while True:
                if isinstance(current_position,np.ndarray):
                    break
                else:
                    time.sleep(1e-2)

        start_time              = time.time()

        mission_waypoints  = np.insert(mission_waypoints,0,
            current_position,axis=0)

        mission_waypoints = self.eliminate_waypoints(
            mission_waypoints)/self.path_scale

        if len(self.obstacles):
            updated_mission_waypoints = self.check_waypoints_obstacles(
                mission_waypoints,obstacles)
            output_path = []

            for i in range(len(updated_mission_waypoints)-1):

                starting_point = updated_mission_waypoints[i]

                goal = updated_mission_waypoints[i+1]

                if self.check_if_collision(starting_point,goal):

                    rrt = smoothed_RRT.RRT(
                            start=starting_point,
                            goal=goal,
                            randArea=[int(-5280/self.path_scale),
                                int(5280/self.path_scale)],
                            obstacleList=list(np.asarray(
                                obstacles)/self.path_scale))
                    path = rrt.Planning(animation=False)

                    smoothedPath = smoothed_RRT.PathSmoothing(path,
                        self.max_iterations,obstacles)

                    rdp_path = rdp.rdp(smoothedPath)

                    output_path = rdp_path

                else:
                    output_path = updated_mission_waypoints

        else:
            output_path = mission_waypoints

        output_path = np.asarray(output_path)*self.path_scale
        return [self.ft2latlng(lat,lng) for (lat,lng) in output_path]