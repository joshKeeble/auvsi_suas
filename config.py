#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
SYSTEM CONFIGURATION VARIABLES
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
import sys
import os

__author__ = "hal112358"

"""
-------------------------------------------------------------------------------
SYSTEM VARIABLES
-------------------------------------------------------------------------------
"""

SYS_NAME = "auvsi_suas"

"""
-------------------------------------------------------------------------------
REGION OF INTEREST TARGETING
-------------------------------------------------------------------------------
"""
#------------------------------------------------------------------------------
# Debugging mode, will check all user args
DEBUG_MODE                  = True
#------------------------------------------------------------------------------
# Will output and display a frame with rois
DISPLAY_ROI                 = True
#------------------------------------------------------------------------------
# Scale of preprocessing resize
# Larger decreases accuracy but increases speed
FRAME_ROI_RESIZE            = 3
#------------------------------------------------------------------------------
# Activate blob detection to find rois
USE_BLOB_DETECTION          = True
#------------------------------------------------------------------------------
#Use Lapacian of Gaussian Blob Detector
L_GAUSSIAN_DETECTION        = False
L_GAUSSIAN_MAX_SIGMA        = 30
L_GAUSSIAN_NUM_SIGMA        = 10
L_GAUSSIAN_THRESHOLD        = 0.1
#------------------------------------------------------------------------------
#Use Difference of Gaussian Blob Detector
D_GAUSSIAN_DETECTION        = False
D_GAUSSIAN_MAX_SIGMA        = 30
D_GAUSSIAN_THRESHOLD        = 0.1
#------------------------------------------------------------------------------
#Use Hessian Blob Detector
HESSIAN_DETECTION           = True
HESSIAN_MAX_SIGMA           = 10
HESSIAN_MIN_SIGMA           = 1
HESSIAN_THRESHOLD           = 0.01

USE_SELECTIVE_SEARCH        = True

# Resize scale for how much to shrink the frame before selective search
# Larger scale will increase speed but decrease accuracy
SELS_RESIZE_SCALE           = 2
SELS_SCALE                  = 600
SELS_SIGMA                  = 0.9
SELS_MIN_SIZE               = 10


#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
"""
-------------------------------------------------------------------------------
REGION OF INTEREST FILTERING
-------------------------------------------------------------------------------
"""
# Filter regions of interest by minimum pixel count
SELS_SIZE_FILTER            = False
# Minimum size of regions of interest
SELS_MIN_ROI_SIZE           = 196


#------------------------------------------------------------------------------
# Filter rois by size compared to size of frame
SELS_RELATIVE_SIZE_FILTER   = True
# Ratio of roi to size of frame
# Smaller values mean that roi will be smaller
SELS_N_PIXEL_RATIO          = 0.3
#------------------------------------------------------------------------------
# Filter by the ratio of side lengths
SELS_SIDE_RATIO_FILTER      = True
# Minimum ratio of sides of region of interest
# Smaller ratios mean rois will be more square
SELS_MIN_SIDE_RATIO         = 2

#------------------------------------------------------------------------------
# Filter overlapping roi by L2 distance
SELS_OVERLAP_FILTER         = False
# Minimum distance between regions of interest
SELS_MIN_DISTANCE           = 20
"""
-------------------------------------------------------------------------------
DATA GENERATION
-------------------------------------------------------------------------------
"""
#------------------------------------------------------------------------------
# Number of generated training images
NUM_GEN_TRAIN_DATA          = 100000
#------------------------------------------------------------------------------
# Number of generated testing images
NUM_GEN_TEST_DATA           = 100000
#------------------------------------------------------------------------------
# Amount that a generated image background will be resized based on
# original size. Larger amounts mean a smaller resize while larger
# amounts mean a smaller resize scale.
GEN_RESIZE_SCALE            = 10
#------------------------------------------------------------------------------
RNDN_SHAPE_RATIO            = 30

#------------------------------------------------------------------------------
# Remove all previously generated data from directories
CLEAN_GEN_DATA              = True
# Number of generated data per class
N_GENERATED_DATA            = 100
"""
-------------------------------------------------------------------------------
COMMUNICATION
-------------------------------------------------------------------------------

	System Overview:
		   						 PAYLOAD      ------ INTEROP
		                            |		  |
		      					    |         |
		MISSION PLANNER ---- GROUND STATION---- 	  GIMBAL
			   |										|
			   ------------------------------------------

	Each connection is over OSC in order to minimize connection error handling

"""
#------------------------------------------------------------------------------
# Mission Planner Host
MISSION_PLANNER_HOST 				= '127.0.0.1'
# Mission Planner Port
MISSION_PLANNER2GROUND_STATION_PORT = 4006

MISSION_PLANNER2GIMBAL_PORT 		= 5006
#------------------------------------------------------------------------------
# Payload Host
GROUND_STATION_HOST 				= '127.0.0.1'
# Ground station to send data to mission planner
# Sends waypoint commands for autonomous flight
GROUND_STATION2MISSION_PLANNER_PORT = 5007
# Ground station to payload port
# Sends commands to change payload modes
GROUND_STATION2PAYLOAD_PORT 		= 5008
#------------------------------------------------------------------------------
GIMBAL_HOST 						= '127.0.0.1'

#------------------------------------------------------------------------------
PAYLOAD_HOST 						= '127.0.0.1'
PAYLOAD2GROUND_STATION 				= 5009

"""
-------------------------------------------------------------------------------
INTEROPERABILITY SYSTEM
-------------------------------------------------------------------------------
"""
INTEROP_USE_ASYNC = False