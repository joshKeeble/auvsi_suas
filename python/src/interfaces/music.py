#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
AUDIO INTERFACE
===============================================================================
Play music because why not?
-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
from pygame import mixer 
import time
import sys
import os

__author__ = "gndctrl2mjrtm"

def play_mp3(mp3_file):
	"""Play a mp3 audio file"""
	if isinstance(mp3_file,str):
		if mp3_file.endswith('.mp3'):
			if os.path.exists(mp3_file):
				start_time = time.time()
				print("Playing music!!")
				mixer.init()
				mixer.music.load(mp3_file)
				mixer.music.play()
				while True:
					pass

def main():
	test_file_path = "/home/hal/Desktop/Ride of the Valkyries by Wagner (Royalty-Free Music).mp3"
	play_mp3(test_file_path)

if __name__ == "__main__":
	main()
