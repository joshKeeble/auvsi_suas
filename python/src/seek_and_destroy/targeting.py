#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
TARGETING
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
from skimage.feature import blob_doh
import numpy as np
import cv2
import sys
import os

try:
    import auvsi_suas.python.src.seek_and_destroy.sel_search as sel_search
    import auvsi_suas.config as config
    
except ModuleNotFoundError as e:
    cwd = os.getcwd().split(os.path.sep)
    project_dir = os.path.sep.join(cwd[:cwd.index("auvsi_suas")])
    print('{}\n\nRun "export PYTHONPATH=$PYTHONPATH:{}"'.format(e,
                project_dir),file=sys.stderr)


__author__ = "hal112358"

"""
===============================================================================

===============================================================================
"""

class Targeting(object):

    #--------------------------------------------------------------------------

    @staticmethod
    def scikit_blob(frame):
        h,w = frame.shape[:2]
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        rois = []
        if config.L_GAUSSIAN_DETECTION:
            blobs_log = blob_log(gray,
                max_sigma=config.L_GAUSSIAN_MAX_SIGMA,
                num_sigma=config.L_GAUSSIAN_NUM_SIGMA,
                threshold=config.L_GAUSSIAN_THRESHOLD)
            blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
            rois.extend(blobs_log)

        if config.D_GAUSSIAN_DETECTION:
            blobs_dog = blob_dog(gray,
                max_sigma=config.D_GAUSSIAN_MAX_SIGMA,
                threshold=config.D_GAUSSIAN_THRESHOLD)
            blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
            rois.extend(blobs_dog)

        if config.HESSIAN_DETECTION:
            blobs_doh = blob_doh(gray,max_sigma=config.HESSIAN_MAX_SIGMA,
                min_sigma=config.HESSIAN_MIN_SIGMA,
                threshold=config.HESSIAN_THRESHOLD)
            rois.extend(blobs_doh)

        candidates = []
        for roi in rois:
            (y,x,r) = roi
            r = 2.5*r
            candidates.append([x,y,2*r,2*r])
        return candidates

    #--------------------------------------------------------------------------

    @staticmethod
    def selective_search(frame):
        frame_height,frame_weight = frame.shape[:2]

        frame = cv2.resize(frame,(int(frame_weight/config.SELS_RESIZE_SCALE),
            int(frame_height/config.SELS_RESIZE_SCALE)))

        process_height,process_weight = frame.shape[:2]
        n_pixels = process_weight*process_height

        img_lbl, regions = sel_search.selective_search(
            frame, scale=config.SELS_SCALE,sigma=config.SELS_SIGMA,
            min_size=config.SELS_MIN_SIZE)

        candidates = []

        for r in regions:
            append_roi = True
            if not (r['rect'] in candidates):
                if config.SELS_RELATIVE_SIZE_FILTER:
                    if ((r['size']/n_pixels > config.SELS_N_PIXEL_RATIO)):
                        append_roi = False

                if append_roi:
                    if config.SELS_SIZE_FILTER:
                        if not (r['size'] > config.SELS_MIN_ROI_SIZE):
                            append_roi = False

                if append_roi:
                    if config.SELS_SIDE_RATIO_FILTER:
                        x,y,w,h = r['rect']
                        if not ((w/max(h,1) < config.SELS_MIN_SIDE_RATIO) and (
                                h/max(w,1) < config.SELS_MIN_SIDE_RATIO)):
                            append_roi = False

                if append_roi:
                    if config.SELS_OVERLAP_FILTER:
                        unique = True
                        if len(candidates):
                            for n in candidates:
                                if (np.linalg.norm(
                                        np.asarray(r['rect'][:2])-np.asarray(
                                        n[:2])) < config.SELS_MIN_DISTANCE):
                                    if (np.linalg.norm(
                                            np.asarray(r['rect'][2:])-np.asarray(
                                            n[2:])) < config.SELS_MIN_DISTANCE):
                                        unique = False
                            if not unique:
                                append_roi = False
                if append_roi:
                    candidates.append(r['rect'])

        candidates = config.SELS_RESIZE_SCALE*np.asarray(
            candidates,dtype=np.int16)

        return candidates

    #--------------------------------------------------------------------------

    def roi_process(self,frame):
        if config.DEBUG_MODE:
            if not isinstance(frame,np.ndarray):
                raise TypeError("Incorrect frame arg type:{}".format(
                    type(frame).__name__))
            if not (len(frame.shape)==3):
                raise Exception("Frame arg is not 3D: n dim: {}".format(
                    len(frame.shape)))

        if not config.USE_BLOB_DETECTION and config.USE_SELECTIVE_SEARCH:
            raise Exception("No ROI method selected")
        h,w = frame.shape[:2]
        r_candidates,s_candidates = None,None
        candidates = []
        if config.DISPLAY_ROI:
            display_frame = frame.copy()

        frame = cv2.resize(frame,(int(w/config.FRAME_ROI_RESIZE),
            int(h/config.FRAME_ROI_RESIZE)))

        if config.USE_BLOB_DETECTION:
            r_candidates = self.scikit_blob(frame)
            candidates.extend(r_candidates)
            if config.DISPLAY_ROI:
                for roi in r_candidates:
                    (x,y,r,_) = list(map(int,roi))
                    r = int(r/2)
                    cv2.circle(display_frame,(x*config.FRAME_ROI_RESIZE,
                        y*config.FRAME_ROI_RESIZE),r*config.FRAME_ROI_RESIZE,
                        [255,0,0],thickness=2,lineType=8,)
            else:
                r_candidates = self.scikit_blob(frame)

        if config.USE_SELECTIVE_SEARCH:
            s_candidates = self.selective_search(frame)
            candidates.extend(s_candidates)
            if config.DISPLAY_ROI:
                for (x,y,w,h) in s_candidates:
                    cv2.rectangle(display_frame,(x*config.FRAME_ROI_RESIZE,
                        y*config.FRAME_ROI_RESIZE),(
                        (x+w)*config.FRAME_ROI_RESIZE,
                        (y+h)*config.FRAME_ROI_RESIZE),
                        [0,255,0])
        
        if config.USE_BLOB_DETECTION:
            r_candidates = np.multiply(r_candidates,config.FRAME_ROI_RESIZE)
        if config.USE_SELECTIVE_SEARCH:
            s_candidates = np.multiply(s_candidates,config.FRAME_ROI_RESIZE)
        if config.DISPLAY_ROI:
            return display_frame,r_candidates,s_candidates
        else:
            return None,r_candidates,s_candidates

    #--------------------------------------------------------------------------

    def video_test(self):
        video_data = cv2.VideoCapture(0)
        while True:
            _,frame = video_data.read()
            print(frame.shape)
            frame = cv2.resize(frame,(1080,952))
            display_frame,candidates = self.roi_process(frame)
            cv2.imshow("frame",display_frame)
            k = cv2.waitKey(1)
            if (k == ord('q')):
                break
        cv2.destroyAllWindows()

    #--------------------------------------------------------------------------

    def video_save_test(self):
        video_data = cv2.VideoCapture(0)
        while True:
            _,frame = video_data.read()
            #print(frame.shape)
            frame = cv2.resize(frame,(1080,952))
            display_frame,candidates = self.roi_process(frame)
            for roi in candidates:
                if not isinstance(roi,np.ndarray):

                    #subframe = frame[]
                    print(roi)
                else:
                    print(roi)
                    #cv2.imshow("roi",roi)
                    #cv2.waitKey(1)
                #cv2.waitKey(0)
            cv2.imshow("frame",display_frame)
            k = cv2.waitKey(1)
            if (k == ord('q')):
                break
        cv2.destroyAllWindows()



    #--------------------------------------------------------------------------

    def frame_test(self):
        frame = cv2.imread("/media/hal/7828-6360/Target-300ft.JPG")
        if config.DISPLAY_ROI:
            display_frame,candidates = self.roi_process(frame)
            h,w = display_frame.shape[:2]
            cv2.imshow("frame",cv2.resize(display_frame,(int(w/1),int(h/1))))
            k = cv2.waitKey(0)
            if (k == ord('q')):
                cv2.destroyAllWindows()

#------------------------------------------------------------------------------

def video_test():
    file_name = '/media/hal/3863-3031/VIDEO/20180602_163921.MOV'
    roi_finder = Targeting()
    if os.path.exists(file_name):
        video_data = cv2.VideoCapture(file_name)
        i = 0
        while True:
            _,frame = video_data.read()

            if not True:
                display_frame,candidates = roi_finder.roi_process(frame)
                cv2.imshow("display_frame",display_frame)
                cv2.waitKey(1)
            i += 1
            if (i>50000):
                cv2.imshow("display_frame",frame)
                cv2.waitKey(1)
        video_data.release()
        cv2.destroyAllWindows()



#------------------------------------------------------------------------------

def main():
    trgt = Targeting()
    trgt.frame_test()
    # trgt.video_save_test()

#------------------------------------------------------------------------------

if __name__ == "__main__":
    #video_test()
    main()
