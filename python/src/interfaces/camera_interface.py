#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
CAMERA INTEFACE
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
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

__author__ = "gndctrl2mjrtm"

"""
===============================================================================
Camera Interface
===============================================================================
"""

class CameraInterface(object):

	def __init__(self):
		self.use_picamera = False
		self.frame = np.ones((config.CAMERA_HEIGHT,config.CAMERA_WIDTH,3),
			dtype=np.uint8)
		self.mode = ''

	#--------------------------------------------------------------------------

	def init_picamera_thread(self):
		self.picamera_thread = threading.Thread(target=self.init_picamera,
			args=())

	#--------------------------------------------------------------------------

	def init_picamera(self):
		"""Initialize picamera video capture"""
		self.mode = 'picamera'
		with picamera.PiCamera() as camera:
		    camera.resolution = (config.CAMERA_WIDTH,config.CAMERA_HEIGHT)
		    camera.framerate = 24
		    time.sleep(2)
		    while True:
			    output = np.empty((config.CAMERA_HEIGHT,config.CAMERA_WIDTH,3),
			    	dtype=np.uint8)
			    camera.capture(output,'rgb')
			    self.frame = output
			    time.sleep(1e-3) # Adjust

	#--------------------------------------------------------------------------

	def init_camera(self):
		""""Initialize the camera"""
		if self.use_picamera:
			self.init_picamera_thread()
		else:
			self.init_cv2()

	#--------------------------------------------------------------------------

	def init_cv2(self,usb_id=0):
		"""Initialize opencv video capture"""
		self.video_data = cv2.VideoCapture(0)
		self.video_data.set(3,3280)
		self.video_data.set(4,2464)
		self.mode = 'cv2'
		

	#--------------------------------------------------------------------------

	def fetch_frame(self):
		"""Return a frame from the video capture method"""
		if (self.mode == ''):
			print("No method specified, automatically using cv2",
				file=sys.stderr)
			self.init_cv2()
			ret,frame = self.video_data.read()

		if (self.mode == 'cv2'):
			ret,frame = self.video_data.read()
			cv2.waitKey(1)

		elif (self.mode == 'picamera'):
			frame = self.frame

		return frame

	#--------------------------------------------------------------------------

	def record_video(self):
		"""Record a video from the video capture method"""
		pass

	#--------------------------------------------------------------------------

	def display_frame(self,frame,title='frame',wait=0):
		"""Display a frame"""
		if not isinstance(frame,np.ndarray):
			raise Exception("Incorrect frame arg type:{}".format(
				type(frame).__name__))
		cv2.imshow(title,frame)
		cv2.waitKey(wait)

"""
===============================================================================

===============================================================================
"""

def main():
	camera = CameraInterface()
	camera.init_cv2()
	print(camera.video_data)
	while True:
		frame = camera.fetch_frame()
		camera.display_frame(frame,wait=1)

if __name__ == "__main__":
	main()
