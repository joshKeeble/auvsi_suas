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
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import numpy as np
import pickle
import random
import copy
import math
import time
import zlib
import cv2
import sys
import os
import re

if (sys.platform == 'win32'):
    sys.path.append('C:\\Users\\soffer\\Desktop')

import auvsi_suas.python.src.communications.mp_communications as mp_coms
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
        self.path_scale         = 30

        # Scale to increase obstacle radius to create buffer zone
        self.obstacle_scale     = 3

        self.current_position   = np.asarray([0,0]) ########################### FIX

        # The waypoints required to be met by the mission
        self.mission_waypoints  = []

        # The obstacles, both moving and stationary
        self.obstacles          = []

        # Display obstacle avoidance
        self.display            = True

        # Previous obstacles
        self.prev_obstacles     = None

        self.prev_path          = None

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
            #ox,oy = self.latlng2ft(n.latitude,n.longitude)
            try:
                radius = n.cylinder_radius
            except:
                radius = n.sphere_radius
            temp_obstacles.append([n.latitude,n.longitude,radius])
        self.obstacles = temp_obstacles

    #--------------------------------------------------------------------------

    def update_waypoints(self,waypoints):
        """Update the waypoints to be sent to path planning"""
        n_waypoints = len(waypoints)

        temp_waypoints = [None for _ in waypoints]
        for n in waypoints:
            #print()
            temp_waypoints[n.order-1] = n

        processed_waypoints = np.zeros((n_waypoints,3))
        for n in temp_waypoints:
            processed_waypoints[n.order-1] = [n.latitude,n.longitude,
            n.altitude_msl]

        self.mission_waypoints = processed_waypoints

    #--------------------------------------------------------------------------

    def set_home_location(self,home_waypoint):
        """Setup the location for ground station"""
        self.home_lat = home_waypoint.latitude
        self.home_lng = home_waypoint.longitude

    #--------------------------------------------------------------------------

    def set_max_altitude(self,max_altitude):
        """Setup the maximum altitude from the mission"""
        self.max_altitude = max_altitude

    #--------------------------------------------------------------------------

    def set_min_altitude(self,min_altitude):
        """Setup the maximum altitude from the mission"""
        self.min_altitude = min_altitude

    #--------------------------------------------------------------------------

    def set_current_position(self,current_position):
        """set the current position of the vehicle"""
        self.current_position = np.asarray(current_position)

    #--------------------------------------------------------------------------

    def set_boundaries(self,boundaries):
        """set the boundary positions"""
        self.boundaries = np.asarray(boundaries)

    #--------------------------------------------------------------------------

    def eliminate_waypoints(self,waypoints):
        """Get rid of the waypoints if close enough"""
        distance = np.linalg.norm(waypoints[1]-waypoints[0])
        if (distance <= self.waypoint_threshold):
            waypoints = waypoints[2:]
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
        if (p1[0] < p2[0]):
            (x1,y1) = p1
            (x2,y2) = p2
            mode = 'f'
        else:
            (x1,y1) = p2
            (x2,y2) = p1
            mode = 'b'

        if not (x2==x1):
            slope = (y2-y1)/(x2-x1)
            for i in range(1,int(x2-x1),stride):
                line.append((x1+i,y1+slope*(i)))
        else:
            min_y = min(y1,y2)
            for i in range(1,abs(int(y2-y1)),stride):
                line.append((x1,min_y+(i)))
        if (mode == 'b'):
            line.reverse()
        return line

    #--------------------------------------------------------------------------

    def check_if_collision(self,p1,p2,obstacles,buffer_zone=1):
        """Check if there is an obstacle in the path"""
        line = self.fetch_line(p1,p2,stride=1)

        collision = False
        for o in obstacles:
            for n in line:
                if (np.linalg.norm(np.asarray(n)-np.asarray(o[:2])) <= o[2]+buffer_zone):
                    collision = True
        return collision

    #--------------------------------------------------------------------------

    def check_waypoints_obstacles(self,mission_waypoints,obstacles,
        moving_obstacle_direction,buffer_zone=1):
        """
        Check for waypoints within obstacles

        Input:
            mission_waypoints: list of (x,y) waypoints
            obstacles: list of (x,y,r) obstacles
            buffer_zone: int of extra radius to create buffer between uav 
                and real obstacle zone
        Output:
            updated_mission_waypoints: waypoints that are not within obstacles

        """
        # Copy of waypoints in case waypoints arg is attribute
        updated_mission_waypoints = mission_waypoints.copy()

        for i,n in enumerate(updated_mission_waypoints):
            for j,o in enumerate(obstacles):

                # If center of waypoint is center of obstacle, point towards
                # next waypoint on exterior of obstacle
                if (np.equal(o[:2],n).all()):

                    # If there is a point next, point towards that waypoint
                    if (i != len(updated_mission_waypoints)-1):
                        if np.equal(moving_obstacle_direction[j],[0.0]).all():
                        #if True:
                            dx      = updated_mission_waypoints[i+1][0]-o[0]
                            dy      = updated_mission_waypoints[i+1][1]-o[1]
                            alpha   = np.arctan(max(dy,1e-3)/max(dx,1e-3))
                        else:

                            v_dx    = moving_obstacle_direction[j][0]
                            v_dy    = moving_obstacle_direction[j][1]
                            v_alpha = np.arctan(max(v_dy,1e-3)/max(v_dx,1e-3))

                            if (v_alpha >= 180):
                                alpha = v_alpha-180
                            else:
                                alpha = v_alpha+180

                        new_x   = o[0]+(o[2]+buffer_zone)*np.cos(alpha)
                        new_y   = o[1]+(o[2]+buffer_zone)*np.sin(alpha)
                        updated_mission_waypoints[i] = [new_x,new_y]

                    # If at end, point towards previous waypoint
                    else:
                        if np.equal(moving_obstacle_direction[j],[0.0]).all():
                        #if True:
                            dx      = updated_mission_waypoints[i+1][0]-o[0]
                            dy      = updated_mission_waypoints[i+1][1]-o[1]
                            alpha   = np.arctan(max(dy,1e-3)/max(dx,1e-3))
                        else:

                            v_dx    = moving_obstacle_direction[j][0]
                            v_dy    = moving_obstacle_direction[j][1]
                            v_alpha = np.arctan(max(v_dy,1e-3)/max(v_dx,1e-3))

                            if (v_alpha >= 180):
                                alpha = v_alpha-180
                            else:
                                alpha = v_alpha+180

                            
                        new_x   = o[0]+(o[2]+buffer_zone)*np.cos(alpha)
                        new_y   = o[1]+(o[2]+buffer_zone)*np.sin(alpha)
                        updated_mission_waypoints[i] = [new_x,new_y]

                # If not in center, move to closest point to waypoint
                elif (np.linalg.norm((n[1]-o[1],n[0]-o[0]))<=o[2]):
                    if np.equal(moving_obstacle_direction[j],[0.0]).all():
                    # if True:
                        dx      = updated_mission_waypoints[i+1][0]-o[0]
                        dy      = updated_mission_waypoints[i+1][1]-o[1]
                        alpha   = np.arctan(max(dy,1e-3)/max(dx,1e-3))
                    else:
                        dx      = updated_mission_waypoints[i+1][0]-o[0]
                        dy      = updated_mission_waypoints[i+1][1]-o[1]
                        #alpha   = np.arctan(max(dy,1e-3)/max(dx,1e-3))
                        v_dx    = moving_obstacle_direction[j][0]
                        v_dy    = moving_obstacle_direction[j][1]
                        v_alpha = np.arctan(max(v_dy,1e-3)/max(v_dx,1e-3))

                        if (v_alpha >= 180):
                            alpha = v_alpha-180
                        else:
                            alpha = v_alpha+180
                            
                    new_x   = o[0]+(o[2]+buffer_zone)*np.cos(alpha)
                    new_y   = o[1]+(o[2]+buffer_zone)*np.sin(alpha)
                    updated_mission_waypoints[i] = [new_x,new_y]

        return np.asarray(updated_mission_waypoints)

    #--------------------------------------------------------------------------

    def apply_geofence(self,current_waypoints,geofence):
        """
        Remove points that are outside of geofence

        Input:
            Waypoints list in x,y
            Geofence list of polygon points in shapely.Polygon
        """
        for i,n in enumerate(current_waypoints):
            current_waypoints[i] = list(n)
        geo_line_points = []
        geo_x,geo_y = geofence.exterior.coords.xy

        geo_points = list(zip(geo_x.tolist(),geo_y.tolist()))
        for i in range(len(geo_points)-1):
            line = self.fetch_line(geo_points[i],geo_points[i+1])
            geo_line_points.extend(list(line))

        # Temporary copy of waypoints in case an attribute is inputed
        temp_waypoints = current_waypoints.copy()
        for i,p in enumerate(current_waypoints):

            # Convert to shapely.Point
            point = Point(p[0],p[1])


            # If the point is outside, remove point
            if not geofence.contains(point):

                lowest = 1e3
                best = -1
                for k,line_point in enumerate(geo_line_points):
                    dist = np.linalg.norm(np.asarray(p[:2])-np.asarray(line_point))
                    if dist < lowest:
                        lowest = dist
                        best = k
                temp_waypoints[i][0] = geo_line_points[best][0]
                temp_waypoints[i][1] = geo_line_points[best][1]
                
                # Find index
                #for j,s in enumerate(temp_waypoints):
                #    if np.equal(s,p).all():
                #        p_index = j
                #        break

                # Remove outside points
                #temp_waypoints = np.delete(temp_waypoints,p_index,0)

        return temp_waypoints

    #--------------------------------------------------------------------------

    def segment_path(self,current_waypoints,primary_waypoints):
        temp_waypoints = current_waypoints.copy()
        temp_primary = primary_waypoints.copy()

        primary_indexes = []

        for p in temp_primary:
            for i,n in enumerate(temp_waypoints):
                if np.equal(n,p).all():
                    primary_indexes.append(i)
                    break

        n_segments = len(primary_indexes)-1
        segments = [[] for _ in range(len(primary_indexes)-1)]

        for i in range(n_segments):
            sel = temp_waypoints[primary_indexes[i]:min(
                primary_indexes[i+1]+1,len(temp_waypoints))]
            for n in sel:
                segments[i].append(n)

        return segments
        '''
        print('-'*80)
        print("temp_waypoints")
        for n in temp_waypoints:
            print(n)
        print("temp_primary")
        for n in temp_primary:
            print(n)
        print('segments')
        for n in segments:
            print(n)
        print('-'*80)
        #'''

    #--------------------------------------------------------------------------

    def create_boundary_points(self,boundary_points):
        """
        
        """
        boundary_line = []
        n_boundary = len(boundary_points)
        for i in range(n_boundary):
            start = boundary_points[i]
            goal = boundary_points[(i+1)%n_boundary]
            if len(boundary_line):
                if not (np.equal(start,boundary_line[-1]).all()):
                    boundary_line.append(start)
            else:
                boundary_line.append(start)

            # Append the points in between
            created_points = self.fetch_line(start,goal,stride=int(100/self.path_scale))

            for j,n in enumerate(created_points):
                boundary_line.append(n)

            # Append the end point
            boundary_line.append(goal)

        return boundary_line

    #--------------------------------------------------------------------------

    def create_altitude_points(self,full_path,primary_waypoints,
        waypoint_altitudes):
        """Add the altitude waypoints to the full path"""
        segments = self.segment_path(full_path,primary_waypoints)
        path_placeholder = np.zeros((len(full_path),3))
        #print('-'*80)
        #print("segments:{}".format(segments))
        #for n in segments:
        #    print(n)
        #print(len(segments))
        #print('-'*80)

        counter = 0
        for i,n in enumerate(segments): 
            segment_range = np.linalg.norm(n[0]-n[-1])
            for j in range(1,len(n)):
                alt = min(self.max_altitude,max(self.min_altitude,
                    (max((np.linalg.norm(n[0]-n[j])/segment_range),0.1)*(
                    waypoint_altitudes[i+1]-(
                    waypoint_altitudes[i])))+waypoint_altitudes[i]))

                path_placeholder[counter][0] = n[j][0]
                path_placeholder[counter][1] = n[j][1]
                path_placeholder[counter][2] = alt
                counter += 1

        #path_placeholder = path_placeholder[:-1]
        path_placeholder = np.asarray(path_placeholder)

        return path_placeholder

    #--------------------------------------------------------------------------

    def partition_path(self,current_waypoints):
        """
        Split up a path into partitions

        Input:
            Current waypoints list of (x,y)
        Output:
            Expanded waypoint list of (x,y)
        """
        # List to make appending easier
        temp_path = []

        # Number of pairs
        steps = len(current_waypoints)-1

        # Cycle through pairs
        for i in range(steps):

            start = current_waypoints[i]
            goal = current_waypoints[i+1]

            # Check if the previous point was starting point
            if len(temp_path):
                if not (np.equal(start,temp_path[-1]).all()):
                    temp_path.append(start)
            else:
                temp_path.append(start)

            # Append the points in between
            created_points = self.fetch_line(start,goal,
                stride=1)

            for j,n in enumerate(created_points):
                temp_path.append(n)

            # Append the end point
            temp_path.append(goal)

        return temp_path

    #--------------------------------------------------------------------------

    def rotate(self,image,angle):
        """Rotate frame"""
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    #--------------------------------------------------------------------------

    def find_path(self,obstacles,mission_waypoints,current_position,boundaries):
        """
        Find the optimal path

        Input:
            obstacles: list of gps coordinates of obstacle [lat,lng,radius]
            mission_waypoints: list of gps waypoints it must cross
            current_position: the current position of the vehicle

        Output:
            output_path: list of gps coordinates of the waypoint path

        """
    
        #sys.exit()

        start_time = time.time()

        current_waypoints = []

        # Conver the waypoints from gps to feet
        for i,n in enumerate(mission_waypoints):
            (x,y) = self.latlng2ft(n[0],n[1])
            current_waypoints.append([x,y,n[2]])

        # Convert the obstacles from gps to feet
        for i,n in enumerate(obstacles):
            (lat,lng) = obstacles[i][:2]
            crds = self.latlng2ft(lat,lng)
            obstacles[i] = [crds[0],crds[1],self.obstacle_scale*n[2]]

        original_obstacles = obstacles.copy()
        current_obstacles = obstacles.copy()

        ####################################################################### TESTING

        # Add the current position to the beginning of the waypoints
        (current_x,current_y) = self.latlng2ft(current_position[0],current_position[1])
        current_p = np.asarray(current_x,current_y,current_position[2])
        current_waypoints = np.insert(current_waypoints,0,
            current_p,axis=0)
        '''
        current_waypoints = np.insert(current_waypoints,
            len(current_waypoints),
            np.asarray((-1000,-2000,200)),
            axis=0)

        current_waypoints = np.insert(current_waypoints,
            len(current_waypoints),
            np.asarray((-1000,2000,400)),
            axis=0)
        '''
        waypoint_altitudes = []
        temp_waypoints = []
        for n in current_waypoints:
            waypoint_altitudes.append(n[2])
            temp_waypoints.append(n[:2])
        current_waypoints = temp_waypoints
        
        boundary_points = []

        fly_zone = boundaries[0].boundary_pts
        for i,n in enumerate(fly_zone):
            (x,y) = self.latlng2ft(n.latitude,n.longitude)
            boundary_points.append((x,y))

        #######################################################################
        
        # Rescale the data down so it can be processed faster
        rescale_down        = lambda z: (np.asarray(z)/self.path_scale)
        original_obstacles  = list(map(rescale_down,original_obstacles))
        current_obstacles   = list(map(rescale_down,current_obstacles))
        current_waypoints   = list(map(rescale_down,current_waypoints))
        boundary_points     = list(map(rescale_down,boundary_points))
        primary_waypoints   = current_waypoints.copy()

        # Create polygon of boundary
        geofence = Polygon(boundary_points)

        full_path = []

        # Number of steps
        


        for i,n in enumerate(current_waypoints):
            if not isinstance(n,list):
                if isinstance(n,np.ndarray):
                    current_waypoints[i] = n.tolist()
                else:
                    current_waypoints[i] = list(n)

        current_waypoints = self.partition_path(current_waypoints)

        current_waypoints = self.apply_geofence(current_waypoints,geofence)

        
        steps = len(current_waypoints)-1

        # Check the waypoints to see if inside obstacle

        # Cycle through steps
        for i in range(steps):
            start = current_waypoints[i]
            goal = current_waypoints[i+1]


            if len(current_obstacles):

                #--------------------------------------------------------------
                # Track moving obstacles
                #--------------------------------------------------------------

                # Placeholder for moving object direction vectors
                moving_obstacle_direction = np.zeros((len(
                    current_obstacles),2))
                dir_obstacles = []

                temp_obstacles = current_obstacles.copy()

                for i,n in enumerate(obstacles):
                    if isinstance(self.prev_obstacles,list):

                        # Obstacle at previous timestamp
                        prev_o = self.prev_obstacles[i]

                        r = np.linalg.norm(np.asarray(prev_o[:2])-np.asarray(n[:2]))
                        moving_obstacle_direction[i][0] = (n[0]-prev_o[0])
                        moving_obstacle_direction[i][1] = (n[1]-prev_o[1])
                        # print(moving_obstacle_direction[i])

                        # Find velocity end point
                        dir_line_point = tuple(list(map(int,np.add(np.asarray(
                            n[:2]),self.path_scale*r*np.asarray(
                            moving_obstacle_direction[i])))))

                        dir_line = self.fetch_line(n[:2],
                            dir_line_point,stride=8)

                        # for d in dir_line:
                            #temp_obstacles.append([d[0]/self.path_scale,d[1]/self.path_scale,n[2]/self.path_scale])
                            # temp_obstacles.append(list(map(rescale_down,d)))

        for i,n in enumerate(current_obstacles):
            if not np.equal(moving_obstacle_direction[i],[0,0]).all():
                current_obstacles[i][2] = 8

        if len(current_obstacles):
            current_waypoints = self.check_waypoints_obstacles(current_waypoints,
                current_obstacles,moving_obstacle_direction)

        for i in range(steps):
            start = current_waypoints[i]
            goal = current_waypoints[i+1]
                # current_obstacles = temp_obstacles
                

            # Check if obstacle in way
            if self.check_if_collision(start,goal,current_obstacles):


############################################################################################################################################################## 
#-------------------------- RDP SECTION CAN GO HERE ----------------------------------------------------------------------------------------------------------
##############################################################################################################################################################
                #print('Start:',start,'\tGoal',goal)
                #print(current_waypoints)
                '''
                rrt = smoothed_RRT.RRT(
                        start=start,
                        goal=goal,
                        randArea=[int(-5280/self.path_scale),
                            int(5280/self.path_scale)],
                        obstacleList=current_obstacles)

                path = rrt.Planning(animation=False)      

                # Smoothen the path
                smoothedPath = smoothed_RRT.PathSmoothing(path,
                    self.max_iterations,obstacles)

                rdp_path = rdp.rdp(smoothedPath)
                print("FULL PATH")

                print(full_path)
                print('*'*80)

                print(rdp_path)
                

                # Fix
                for p in rdp_path:
                    if len(full_path):
                        if not (np.equal(p,full_path[-1]).all()):
                            full_path.append(p)
                    else:
                        full_path.append(p)
                print('*'*80)
                print("FULL PATH")
                print(full_path)
                '''
##############################################################################################################################################################

                if len(full_path):
                    if not (np.equal(start,full_path[-1]).all()):
                        full_path.append(list(start))
                else:
                    full_path.append(list(start))

                full_path.append(list(goal))
                #'''
            else:
                if len(full_path):
                    if not (np.equal(start,full_path[-1]).all()):
                        full_path.append(list(start))
                else:
                    full_path.append(list(start))

                full_path.append(list(goal))

        segmented_path = self.segment_path(current_waypoints,primary_waypoints)

        
        for i,n in enumerate(segmented_path):
            segmented_path[i] = rdp.rdp(segmented_path[i])

        path_placeholder = []
        for i,n in enumerate(segmented_path):
            for p in n:
                path_placeholder.append(list(p))
        # print(full_path)
        # print(path_placeholder)
        full_path = path_placeholder

        # print(primary_waypoints)
        primary_indexes = np.zeros((len(primary_waypoints)))
        for j,checkpoint in enumerate(primary_waypoints):
            distance = 1e4
            index = -1
            for i,n in enumerate(full_path):
                waypoint_distance = np.linalg.norm(np.subtract(checkpoint,n))
                if waypoint_distance<distance:
                    distance = waypoint_distance
                    index = i
            primary_indexes[j] = index
        # print(primary_indexes)

        #sys.exit()

        self.prev_path = full_path

        rescale_up = lambda z: (np.asarray(z)*self.path_scale)
        original_obstacles = list(map(rescale_up,original_obstacles))
        current_obstacles = list(map(rescale_up,current_obstacles))
        current_waypoints = list(map(rescale_up,current_waypoints))
        primary_waypoints = list(map(rescale_up,primary_waypoints))
        boundary_points = list(map(rescale_up,boundary_points))
        full_path = list(map(rescale_up,full_path))

        #print(full_path,primary_waypoints,waypoint_altitudes)
        full_path = self.create_altitude_points(full_path,
            primary_waypoints,waypoint_altitudes)

        self.prev_obstacles = original_obstacles


        if self.display:

            display_size = 8000

            display_path = full_path.copy()
            display_path = np.insert(display_path,0,current_position[:3],axis=0)
            
            display_scale = 10
            frame = np.ones((int(display_size/display_scale),
                int(display_size/display_scale),3),dtype=np.uint8)

            if len(current_obstacles):
                for i,o in enumerate(original_obstacles):
                    center = (int((o[0]+(display_size/2))/display_scale),
                        int(((display_size/2)-o[1])/display_scale))
                    r = int(o[2]/display_scale)

                    dir_line = (center[0]+10*moving_obstacle_direction[i][0],
                        center[1]-10*moving_obstacle_direction[i][1],)
                    dir_line = tuple(map(int,dir_line))

                    cv2.line(frame,center,dir_line,(0,0,255),3)

                    cv2.circle(frame,center,r,(0,255,0),thickness=3,
                        lineType=8,shift=0)
                #--------------------------------------------------------------

                for i,o in enumerate(current_obstacles):
                    center = (int((o[0]+(display_size/2))/display_scale),
                        int(((display_size/2)-o[1])/display_scale))
                    r = int(o[2]/display_scale)

                    cv2.circle(frame,center,r,(0,255,0),thickness=2,
                        lineType=8,shift=0)

                    r = int(o[2]/(display_scale*self.obstacle_scale))

                    cv2.circle(frame,center,r,(0,0,255),thickness=4,
                        lineType=8,shift=0)

            #------------------------------------------------------------------
            #'''
            n_boundary = len(boundary_points)
            for i in range(n_boundary):
                start = boundary_points[i]
                goal = boundary_points[(i+1)%n_boundary]
                (x1,y1) = start
                (x2,y2) = goal
                (x1,y1) = (int((x1+(display_size/2))/display_scale),
                    int(((display_size/2)-y1)/display_scale))
                (x2,y2) = (int((x2+(display_size/2))/display_scale),
                    int(((display_size/2)-y2)/display_scale))
                cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),
                    (255,255,255),1)

            #'''
            #------------------------------------------------------------------

            for i,w in enumerate(current_waypoints):
                center = (int((w[0]+(display_size/2))/display_scale),
                    int(((display_size/2)-w[1])/display_scale))
                r = 3
                cv2.circle(frame,center,r,(255,0,0),thickness=3,lineType=8,shift=0)

            #------------------------------------------------------------------

            for i,w in enumerate(primary_waypoints):
                center = (int((w[0]+(display_size/2))/display_scale),
                    int(((display_size/2)-w[1])/display_scale))
                r = 4
                cv2.circle(frame,center,r,(255,255,0),thickness=3,lineType=8,shift=0)

            #------------------------------------------------------------------
        
            for i in range(len(display_path)-1):
                (x1,y1,_) = display_path[i]
                (x2,y2,_) = display_path[i+1]
                
                (x1,y1) = (int((x1+(display_size/2))/display_scale),
                    int(((display_size/2)-y1)/display_scale))
                (x2,y2) = (int((x2+(display_size/2))/display_scale),
                    int(((display_size/2)-y2)/display_scale))

                cv2.line(frame,(x1,y1),(x2,y2),(255,255,255),1)
            
            cv2.imshow("frame",self.rotate(np.flip(frame,1),90))
            cv2.waitKey(0)

        if self.display:
            cv2.destroyAllWindows()

        output_path = []
        for (x,y,h) in full_path:
            (lat,lng) = self.ft2latlng(x,y)
            output_path.append([lat,lng,h])

        return output_path

"""
===============================================================================

===============================================================================
"""

class ClientBaseType(object):
    """ ClientBaseType is a simple base class which provides basic functions.

    The attributes are obtained from the 'attrs' property, which should be
    defined by subclasses.
    """

    # Subclasses should override.
    attrs = []

    def __eq__(self, other):
        """Compares two objects."""
        for attr in self.attrs:
            if self.__dict__[attr] != other.__dict__[attr]:
                return False
        return True

    def __repr__(self):
        """Gets string encoding of object."""
        return "%s(%s)" % (self.__class__.__name__,
                           ', '.join('%s=%s' % (attr, self.__dict__[attr])
                                     for attr in self.attrs))

    def __unicode__(self):
        """Gets unicode encoding of object."""
        return unicode(self.__str__())

    def serialize(self):
        """Serialize the current state of the object."""
        serial = {}
        for attr in self.attrs:
            data = self.__dict__[attr]
            if isinstance(data, ClientBaseType):
                serial[attr] = data.serialize()
            elif isinstance(data, list):
                serial[attr] = [d.serialize() for d in data]
            elif data is not None:
                serial[attr] = data
        return serial

    @classmethod
    def deserialize(cls, d):
        """Deserialize the state of the object."""
        if isinstance(d, cls):
            return d
        else:
            return cls(**d)

class Waypoint(ClientBaseType):
    """Waypoint consisting of an order, GPS position, and optional altitude.

    Attributes:
        order: An ID giving relative order in a set of waypoints.
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        altitude: Optional. Altitude in feet MSL.

    Raises:
        ValueError: Argument not convertable to int or float.
    """

    attrs = ['order', 'latitude', 'longitude', 'altitude_msl']

    def __init__(self, order, latitude, longitude, altitude_msl=None):
        self.order = int(order)
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.altitude_msl = None
        if altitude_msl is not None:
            self.altitude_msl = float(altitude_msl)


class FlyZone(ClientBaseType):
    """Flight boundary consisting of GPS polygon and altitude range.

    Attributes:
        boundary_pts: List of Waypoint defining a polygon.
        altitude_msl_min: Minimum altitude in feet MSL.
        altitude_msl_max: Maximum altitude in feet MSL.

    Raises:
        ValueError: Argument not convertable to float.
    """

    attrs = ['boundary_pts', 'altitude_msl_min', 'altitude_msl_max']

    def __init__(self, boundary_pts, altitude_msl_min, altitude_msl_max):
        self.boundary_pts = [Waypoint(0,bp[0],bp[1]) for bp in boundary_pts]
        self.altitude_msl_min = float(altitude_msl_min)
        self.altitude_msl_max = float(altitude_msl_max)

host = 'localhost'
port = 4023


def main():
    #gs2mp_client = osc_client.OSCClient('192.168.1.42',5005)
    #gs2mp_client.init_client()

    stealth = StealthMode()
    obstacles = [(45.9616069,-121.2746859,50)]
    mission_waypoints = [(45.9626361,-121.2751794,200)]

    current_position = (45.960615,-121.274267,0)
    home_waypoint = Waypoint(1,current_position[0],current_position[1])

    stealth.set_home_location(home_waypoint)
    stealth.set_max_altitude(400)
    stealth.set_min_altitude(100)

    print(stealth.latlng2ft(mission_waypoints[0][0],mission_waypoints[0][1]))

    boundary_points = [(45.9601301,-121.2709951),(45.9598616,-121.2774324),(45.9643066,-121.2795138),(45.9647988,-121.2711239)]
    boundary = FlyZone(boundary_points,100,400)
    path = stealth.find_path(obstacles,mission_waypoints,current_position,[boundary])
    print('Path:',path)


    while True:
        mp_client = mp_coms.MissionPlannerClient(host,port)
        #print("sending:{}\n{}".format(path,'-'*80))
        #path = np.asarray(path)
        #path = pickle.dumps(path)
        #path = zlib.compress(path)
        #path = [1,2,3,4,5]
        path = np.asarray(np.reshape(path,(1,-1))[0],dtype=np.float32).tolist()

        #path = [1,2,3,4,5]
        print(path)

        mp_client.send_data(path)
       # mp_client.client_socket.close()
        #gs2mp_client.send_data(path,channel='path')
        time.sleep(1)
        mp_client.client_socket.close()
        break


if __name__ == "__main__":
    main()
