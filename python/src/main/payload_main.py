#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
AUVSI SUAS PAYLOAD MAIN
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
import tensorflow as tf
import numpy as np 
import threading
import cv2
import sys
import os

try:
    from mvnc import mvncapi as mvnc
except:
    print("Running without Movidius NCS",file=sys.stderr)

#try:
import auvsi_suas.python.src.seek_and_destroy.neural_processing as classifier
import auvsi_suas.python.src.seek_and_destroy.preprocessing as preprocessing
import auvsi_suas.python.src.seek_and_destroy.fetch_color as fetch_color
import auvsi_suas.python.src.seek_and_destroy.fetch_gps as fetch_gps
import auvsi_suas.python.src.seek_and_destroy.targeting as targeting
import auvsi_suas.python.src.communications.osc_client as osc_client
import auvsi_suas.python.src.communications.osc_server as osc_server
import auvsi_suas.python.src.interfaces.camera_interface as camera
import auvsi_suas.config as config


class PayloadMain(object):

    def __init__(self):
        self.use_picamera = False
        self.frame = -1

    def init_camera(self):
        self.camera_interface = camera.CameraInterface()

    def init_server(self):
        pass

    #--------------------------------------------------------------------------

    def init_classification(self):
        shapeNet = classifier.SmallAlexNet(None,None,None,None,
            mode='inference',name='shapeNet')
        shapeNet.init_network()


        with tf.Session() as sess:
            # Initialize variables
            sess.run(shapeNet.init)

            # Restore model weights from previously saved model
            # cnn.saver.restore(sess,cnn.model_path)



            targeting_functions = targeting.Targeting()
            target_preprocessing = preprocessing.TargetProcessing()

            while True:
                original_frame = self.frame.copy()
                or_height,or_width = original_frame.shape[:2]
                rescale = 2
                frame = cv2.resize(original_frame,(int(or_width/rescale),
                    int(or_height/rescale)))
                height,width = frame.shape[:2]

                display_frame,r_candidates,s_candidates = targeting_functions.roi_process(frame)
                cv2.imshow("frame",display_frame)

                if isinstance(r_candidates,np.ndarray):
                    for roi in r_candidates:
                        if isinstance(roi,(np.ndarray,list)):
                            try:
                                #(x,y,w,h) = list(map(int,roi))
                                #print(x,y,w,h)
                                #print(y,min(y+h,height-1),x,min(x+w,width-1))
                                #region = frame[y:min(y+h,height-1),x:min(x+w,width-1)]


                                region = cv2.imread('/home/hal/Desktop/Programming/generated_data/letters/B/circle_B_193.jpg')
                                #cv2.imshow('region',region)
                                region = target_preprocessing.process_shape_frame(region)
                                #shape_pred = shapeNet.evaluate_frame(sess,frame)
                                #if (pred != 'Noise'):
                                    #letter_pred = letterNet.evaluate_frame(
                                    #    sess,frane) ##### FIX WITH OCR
                                    #if (letter_pred != 'Noise'):
                                    #    gps = fetch_gps.estimate_gps(x+(w/2),y+(h/2))
                                    #    colors = 
                                cv2.imshow('region',region)
                                cv2.waitKey(1)
                            except Exception as e:
                                exc_type,exc_obj,exc_tb = sys.exc_info()
                                fname = os.path.split(
                                    exc_tb.tb_frame.f_code.co_filename)[1]
                                print("Classification Error:",exc_type,fname,
                                    exc_tb.tb_lineno,e,
                                    file=sys.stderr)
                        else:
                            pass

                if isinstance(s_candidates,np.ndarray):
                    for roi in s_candidates:
                        if isinstance(roi,(np.ndarray,list)):
                            try:
                                #(x,y,w,h) = list(map(int,roi))
                                #print(x,y,w,h)
                                #print(y,min(y+h,height-1),x,min(x+w,width-1))
                                #region = frame[y:min(y+h,height-1),x:min(x+w,width-1)]


                                region = cv2.imread('/home/hal/Desktop/Programming/generated_data/letters/B/circle_B_193.jpg')
                                #cv2.imshow('region',region)
                                region = target_preprocessing.process_shape_frame(region)
                                #shape_pred = shapeNet.evaluate_frame(sess,frame)
                                #if (pred != 'Noise'):
                                    #letter_pred = letterNet.evaluate_frame(
                                    #    sess,frane) ##### FIX WITH OCR
                                    #if (letter_pred != 'Noise'):
                                    #    gps = fetch_gps.estimate_gps(x+(w/2),y+(h/2))
                                    #    colors = 
                                cv2.imshow('region',region)
                                cv2.waitKey(1)
                            except Exception as e:
                                exc_type,exc_obj,exc_tb = sys.exc_info()
                                fname = os.path.split(
                                    exc_tb.tb_frame.f_code.co_filename)[1]
                                print("Classification Error:",exc_type,fname,
                                    exc_tb.tb_lineno,e,
                                    file=sys.stderr)
                        else:
                            pass

                
                k = cv2.waitKey(1)
                if (k == ord('q')):
                    break
            cv2.destroyAllWindows()

    #--------------------------------------------------------------------------

    def activate_classifier(self):
        classifier_thread = threading.Thread(target=self.init_classification,args=())
        classifier_thread.daemon = True
        classifier_thread.start()


    def main(self):
        self.init_camera()
        self.activate_classifier()
        while True:
            self.frame = self.camera_interface.fetch_frame()
            #cv2.imshow("frame",self.frame)
            #cv2.waitKey(1)



def main():
    payload = PayloadMain()
    payload.main()


if __name__ == "__main__":
    main()