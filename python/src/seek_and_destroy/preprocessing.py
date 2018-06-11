#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
AUVSI SUAS IMAGE PROCESSING
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
import numpy as np 
import cv2
import sys
import os

"""
===============================================================================
Image Preprocessing
===============================================================================
"""

class FrameProcessing(object):

    def __init__(self):
        pass

"""
===============================================================================
Image Preprocessing
===============================================================================
"""

class TargetProcessing(object):

    def __init__(self):
        self.criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

    #--------------------------------------------------------------------------

    def k_cluster(self,frame,k=4):
        """K-Clustering frame for k colors"""
        frame_shape = frame.shape
        frame       = frame.reshape((-1,3))
        frame       = np.float32(frame)
        ret,label,center = cv2.kmeans(frame,k,None,self.criteria,10,
            cv2.KMEANS_RANDOM_CENTERS)
        center      = np.uint8(center)
        res         = center[label.flatten()]
        res2        = res.reshape((frame_shape))
        return res2

    #--------------------------------------------------------------------------

    def process_shape_frame(self,frame):
        """Preprocess the frame for shapes"""
        if not (np.equal(frame.shape,np.asarray([100,100,3])).all()):
            frame = cv2.resize(frame,(100,100))
        frame = self.k_cluster(frame,k=2)
        frame = np.reshape(frame,(100,100,3))
        frame = cv2.resize(frame,(50,50))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,115,1)
        frame = cv2.Canny(frame,50,150)
        return frame

    #--------------------------------------------------------------------------

    def frame2tensor(self,frame):
        """Prepare a frame to be evaluated by Tensorflow"""
        return np.expand_dims(frame,axis=2)/255.