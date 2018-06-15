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

    def process_letter_frame(self,frame):
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        frame = self.k_cluster(frame,k=5)

        cv2.imshow("letter_frame",frame)



    #--------------------------------------------------------------------------

    def process_shape_frame(self,frame):
        """Preprocess the frame for shapes"""
        #if not (np.equal(frame.shape,np.asarray([100,100,3])).all()):
        #    frame = cv2.resize(frame,(100,100))
        #frame = self.k_cluster(frame,k=2)
        #frame = np.reshape(frame,(100,100,3))
        frame = cv2.resize(frame,(50,50))
        #cv2.imshow('original',frame)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        frame = self.k_cluster(frame,k=2)
        #cv2.imshow("k-frame",frame)
        h,s,v = cv2.split(frame)

        shape_color = np.reshape(v,(-1,1))[-1]
        background = np.reshape(v,(-1,1))[0]
        for n in np.reshape(v,(-1,1)):
            if n != background:
                shape_color = n
                break

        #print(shape_color)
        #print(background)
        #ret,v = cv2.threshold(v,background,shape_color,cv2.THRESH_BINARY)
        #v = cv2.inRange(frame,background,shape_color)

        #for n in v:
        #    print(n)

        #frame = cv2.cvtColor(frame,cv2.COLOR_HSV2BGR)
        #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #v = cv2.adaptiveThreshold(v,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #    cv2.THRESH_BINARY,115,1)
        v = cv2.Canny(frame,min(background,shape_color)-1,max(background,shape_color))
        v = 255-v
        #cv2.imshow('v',v)
        return v

    #--------------------------------------------------------------------------

    def frame2tensor(self,frame):
        """Prepare a frame to be evaluated by Tensorflow"""
        return np.expand_dims(frame,axis=2)/255.