#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
AUVSI SUAS MAIN
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
import tensorflow as tf
import numpy as np 
import cv2
import sys
import os

#try:
import auvsi_suas.python.src.seek_and_destroy.neural_processing as classifier
import auvsi_suas.python.src.seek_and_destroy.fetch_color as fetch_color
import auvsi_suas.python.src.seek_and_destroy.fetch_gps as fetch_gps
import auvsi_suas.python.src.seek_and_destroy.targeting as targeting
import auvsi_suas.python.src.communications.osc_client as osc_client
import auvsi_suas.python.src.communications.osc_server as osc_server
import auvsi_suas.python.src.interfaces.camera_interface as camera

import auvsi_suas.config as config
'''  
except ModuleNotFoundError as e:
    cwd = os.getcwd().split(os.path.sep)
    project_dir = os.path.sep.join(cwd[:cwd.index("auvsi_suas")])
    print('{}\n\nRun "export PYTHONPATH=$PYTHONPATH:{}"'.format(e,
                project_dir),file=sys.stderr)
'''
"""
===============================================================================
Seek and Destory UAV Main
===============================================================================
"""

def handle(data): #------------------------------------------------------------ FIX
    print(data)


class SeekAndDestroyUAV(object):

    def __init__(self):
        #self.payload_osc_group = config.PAYLOAD_OSC_GROUP
        #self.payload_osc_port  = config.PAYLOAD2GROUND_STATION_PORT
        #self.gnd_osc_group     = config.GROUND_STATION_HOST
        #self.gnd_osc_port      = config.GROUND_STATION_OSC_PORT
        pass

    #--------------------------------------------------------------------------

    def init_osc_client(self):
        """Initialize OSC client to send data to ground station"""
        self.payload2gs_client = osc_client.OSCCLient(self.payload_osc_group,
            self.payload_osc_port)
        self.payload2gs_client.init_client()

    #--------------------------------------------------------------------------

    def init_osc_server(self):
        """Initialize OSC server to recieve data from the ground station"""
        self.gs2payload_server = osc_server.OSCServer(self.gnd_osc_group,
            self.gnd_osc_port)
        self.gs2payload_server.init_server(handle)

    #--------------------------------------------------------------------------

    def main(self):
        #self.init_osc_server()
        #self.init_osc_client()

        shapeNet = classifier.SmallAlexNet(None,None,None,None,mode='inference',name='shapeNet')
        shapeNet.init_network()


        with tf.Session() as sess:
            # Initialize variables
            sess.run(shapeNet.init)

            # Restore model weights from previously saved model
            # cnn.saver.restore(sess,cnn.model_path)



            targeting_functions = targeting.Targeting()
            target_preprocessing = preprocessing.TargetProcessing()

            video_data = camera.CameraInterface()
            video_data.init_cv2()

            while True:
                frame = video_data.fetch_frame()
                frame = cv2.imread()
                height,width = frame.shape[:2]
                #print(frame.shape)
                #frame = cv2.resize(frame,(1080,952))
                display_frame,r_candidates,s_candidates = targeting_functions.roi_process(frame)
                cv2.imshow("frame",display_frame)
                for roi in s_candidates:
                    if isinstance(roi,(np.ndarray,list)):
                        try:
                        #print(roi)
                            (x,y,w,h) = list(map(int,roi))
                            print(x,y,w,h)
                            print(y,min(y+h,height-1),x,min(x+w,width-1))
                            region = frame[y:min(y+h,height-1),x:min(x+w,width-1)]
                            region = target_preprocessing.process_shape_frame(frame)
                            shape_pred = shapeNet.evaluate_frame(sess,frame)
                            if (pred != 'Noise'):
                                letter_pred = letterNet.evaluate_frame(sess,frane) ##### FIX WITH OCR
                                if (letter_pred != 'Noise'):
                                    gps = fetch_gps.estimate_gps(x+(w/2),y+(h/2))
                                    colors = 
                            cv2.imshow('region',region)
                            cv2.waitKey(1)
                        except:
                            pass
                    else:
                        pass
                
                k = cv2.waitKey(1)
                if (k == ord('q')):
                    break
            cv2.destroyAllWindows()

"""
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

"""

if __name__ == "__main__":
    SeekAndDestroyUAV().main()