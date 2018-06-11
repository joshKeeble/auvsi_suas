#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
TARGETING TEST
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
import numpy as np
import cv2
import sys
import os

import auvsi_suas.config as config
import targeting


fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

skip_frames = 4900

end_frame = 8000

frame_interval = 2

output_size = (640,480)

VIDEO_FILE_PATH = "/Users/rosewang/Desktop/flight_test.mov"

cap = cv2.VideoCapture(VIDEO_FILE_PATH)

trgt = targeting.Targeting()
frame_count = 0
while(cap.isOpened()):
    frame_count += 1
    # print(frame_count)
    ret, frame = cap.read()
    # print(frame_count%frame_interval)
    if frame_count > skip_frames and not (frame_count%frame_interval):
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame,candidates = trgt.roi_process(frame)
        # cv2.imshow('frame',frame)
        out.write(cv2.resize(frame,output_size))

        if (frame_count > end_frame):
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

out.release()
cap.release()
cv2.destroyAllWindows()