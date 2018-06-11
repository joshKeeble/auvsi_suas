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

try:
    from mvnc import mvncapi as mvnc
except:
    print("Running without Movidius NCS",file=sys.stderr)

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

    def tf_main(self):
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

    #--------------------------------------------------------------------------

    def can_connect_ncs(self):
        """Boolean function for if there is a ncs connected"""
        return 1 if len(mvnc.enumerate_devices()) else 0

    #--------------------------------------------------------------------------

    def movidius_main(self):
        """Run Seek and Destroy with Intel Movidius NCS"""
        graph_filename = './shapeNet_graph'

        devices = mvnc.enumerate_devices()
        
        if not self.can_connect_ncs():
            print("Warning, NCS not connected, runnning on Tensorflow instead",
                file=sys.stderr)
            self.tf_main()
        else:
            device = mvnc.Device(devices[0])
            device.open()

            #Load graph
            with open(graph_filename,mode='rb') as graph_file:
                pretrained_graph = graph_file.read()

            #Load preprocessing data
            mean = 128 
            std = 1/128 

            #Load categories
            categories = []
            with open('../seek_and_destroy/shapes.txt','r') as f:
                for line in f:
                    cat = line.split('\n')[0]
                    if cat != 'classes':
                        categories.append(cat)
                f.close()
                print('Number of categories:', len(categories))

            #Load image size
            with open(path_to_networks + 'inputsize.txt', 'r') as f:
                reqsize = int(f.readline().split('\n')[0])

            graph = mvnc.Graph('graph')
            fifoIn, fifoOut = graph.allocate_with_fifos(device, pretrained_graph)

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
                            (x,y,w,h) = list(map(int,roi))
                            print(x,y,w,h)
                            print(y,min(y+h,height-1),x,min(x+w,width-1))
                            frame = target_preprocessing.process_shape_frame(frame)
                            region = frame[y:min(y+h,height-1),x:min(x+w,width-1)]
                            
                            region = region.astype(numpy.float32)

                            dx,dy,dz= region.shape
                            delta=float(abs(dy-dx))
                            if dx > dy: #crop the x dimension
                                region=region[int(0.5*delta):dx-int(0.5*delta),0:dy]
                            else:
                                region=region[0:dx,int(0.5*delta):dy-int(0.5*delta)]
                                
                            region = cv2.resize(region,(reqsize, reqsize))

                            region = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

                            for i in range(3):
                                region[:,:,i] = (region[:,:,i]-mean)*std

                            print('Start download to NCS...')
                            start_time = time.time()
                            graph.queue_inference_with_fifo_elem(fifoIn, fifoOut,region,'user object')
                            output, userobj = fifoOut.read_elem()

                            top_inds = output.argsort()[::-1][:5]

                            print(''.join(['*' for i in range(79)]))
                            print('inception-v1 on NCS')
                            print(''.join(['*' for i in range(79)]))
                            for i in range(5):
                                print(top_inds[i], categories[top_inds[i]], output[top_inds[i]])

                            print(''.join(['*' for i in range(79)]))
                            print("time elapsed:{}".format(time.time()-start_time))
                            '''
                            if (pred != 'Noise'):
                                letter_pred = letterNet.evaluate_frame(sess,frane) ##### FIX WITH OCR
                                if (letter_pred != 'Noise'):
                                    gps = fetch_gps.estimate_gps(x+(w/2),y+(h/2))
                                    colors = '''
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

            fifoIn.destroy()
            fifoOut.destroy()
            graph.destroy()
            device.close()
            print('Finished')


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