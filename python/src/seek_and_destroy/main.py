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
from neural_processing import TFConvNN
from targeting import Targeting
import tensorflow as tf
import cv2
import sys
import os

import auvsi_suas.config as config

__author__ = "hal112358"


class UAVTargeting(object):

    def __init__(self):
        self.host = config.PAYLOAD_HOST


class GroundStationTargeting(object):

    def __init__(self):
        self.osc_group = config.TARGETING_OSC_GROUP
        self.osc_port = config.TARGETING_OSC_PORT
        

def main():
    cnn  = TFConvNN()
    trgt = Targeting()
    cnn.init_network()

    with tf.Session() as sess:
        # Initialize variables
        sess.run(cnn.init)

        # Restore model weights from previously saved model
        cnn.saver.restore(sess,cnn.model_path)
        video_data = cv2.VideoCapture(0)
        while True:
            _,frame = video_data.read()
            display_frame,candidates = trgt.roi_process(frame)
            for c in candidates:
                if (len(c)==4):
                    (x,y,w,h) = c
                    subframe = frame[int(x):int(x+w),
                        int(y):int(y+h)]
                    h,w = subframe.shape[:2]
                    if (w*h):
                        cnn.evaluate_frame(sess,subframe)
                        if config.DISPLAY_ROI:
                            cv2.imshow("subframe",subframe)
                            cv2.waitKey(1)

            # cv2.imshow("frame",display_frame)
            # cnn.evaluate_frame(sess,frame)
            # cv2.waitKey(1)

if __name__ == "__main__":
    main()
