#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
SELECECTIVE SEARCH
===============================================================================
File Description
-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
import numpy as np
import skimage
import cv2
import sys
import os

__author__ = ("AlpacaDB","hal112358")

"""
===============================================================================
SELECECTIVE SEARCH CLASS
===============================================================================
"""

class SelectiveSearch(object):

    def __init__(self):
        self.color_bins = 25
        self.texture_bins = 10

    #--------------------------------------------------------------------------

    def generate_segments(im_orig,scale,sigma,min_size):
        """segment smallest regions by Felzenswalb and Huttenlocher"""
        im_mask = skimage.segmentation.felzenszwalb(skimage.util.img_as_float(
            im_orig),scale=scale,sigma=sigma,min_size=min_size)

        im_orig = np.append(im_orig,numpy.zeros(
            im_orig.shape[:2])[:,:,np.newaxis],axis=2)

        im_orig[:,:,3] = im_mask
        return im_orig

    #--------------------------------------------------------------------------

    def sim_color(self,r1,r2):
        """Calculate sum of color histogram intersection"""
        return sum([min(a,b) for a,b in zip(r1["hist_c"],r2["hist_c"])])

    #--------------------------------------------------------------------------

    def sim_texture(self,r1,r2):
        """Calculate sum of texture historgram intersection"""
        return sum([min(a,b) for a,b in zip(r1["hist_t"],r2["hist_t"])])

    #--------------------------------------------------------------------------

    def sim_size(self,r1,r2,im_size):
        """Calculate size similarity over the image"""
        return (1.0-(r1["size"]+r2["size"])/im_size)

    #--------------------------------------------------------------------------

    def sim_fill(self,r1,r2,im_size):
        """Calculate fill similarity over the image"""
        size = (
            (max(r1["max_x"],r2["max_x"]) - min(r1["min_x"],r2["min_x"]))
            * (max(r1["max_y"],r2["max_y"]) - min(r1["min_y"],r2["min_y"]))
        )
        return (1.0-(size-r1["size"]-r2["size"])/im_size)

    #--------------------------------------------------------------------------

    def calc_similarity(self,r1,r2,im_size):
        """Calculate similarity between regions"""
        s = (self.sim_color(r1,r2)+self.sim_texture(r1,r2)+self.sim_size(
            r1,r2,im_size)+self.sim_fill(r1,r2,im_size))
        return s

    #--------------------------------------------------------------------------

    def calc_color_hist(self,img):
        """calculate colour histogram for each region"""
        hist = np.array([])
        for colour_channel in (0, 1, 2):
            c    = img[:, colour_channel]
            hist = np.concatenate([hist]+[np.histogram(
                c,self.color_bins,(0.0,255.0))[0]])
        return (hist/len(img))

    #--------------------------------------------------------------------------

    def calc_texture_gradient(self,img):
        """
            calculate texture gradient for entire image

            The original SelectiveSearch algorithm proposed Gaussian derivative
            for 8 orientations, but we use LBP instead.

            output will be [height(*)][width(*)]
        """
        ret = np.zeros((img.shape[0],img.shape[1],img.shape[2]))
        for colour_channel in (0, 1, 2):
            ret[:,:,colour_channel] = skimage.feature.local_binary_pattern(
                img[:,:,colour_channel], 8, 1.0)
        return ret

    #--------------------------------------------------------------------------

    def calc_texture_hist(img):
        """
            calculate texture histogram for each region

            calculate the histogram of gradient for each colours
            the size of output histogram will be
                BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
        """
        hist = np.array([])
        for colour_channel in (0, 1, 2):

            # mask by the colour channel
            fd = img[:,colour_channel]

            # calculate histogram for each orientation and concatenate them all
            # and join to the result
            hist = np.concatenate(
                [hist]+[np.histogram(fd,self.texture_bins,(0.0,1.0))[0]])

        return (hist/len(img))

    #--------------------------------------------------------------------------

    def _extract_regions(img):

        regions = {}

        # get hsv image
        #hsv = skimage.color.rgb2hsv(img[:, :, :3])

        hsv = cv2.cvtColor(img[:,:,:3],cv2.COLOR_RGB2HSV)

        # pass 1: count pixel positions
        for y, i in enumerate(img):
            for x, (r, g, b, l) in enumerate(i):

                # initialize a new region
                if l not in regions:
                    regions[l] = {
                        "min_x": 0xffff, "min_y": 0xffff,
                        "max_x": 0, "max_y": 0, "labels": [l]}

                # bounding box
                if regions[l]["min_x"] > x:
                    regions[l]["min_x"] = x

                if regions[l]["min_y"] > y:
                    regions[l]["min_y"] = y

                if regions[l]["max_x"] < x:
                    regions[l]["max_x"] = x

                if regions[l]["max_y"] < y:
                    regions[l]["max_y"] = y

        # pass 2: calculate texture gradient
        tex_grad = self.calc_texture_gradient(img)

        # pass 3: calculate colour histogram of each region
        for k,v in list(regions.items()):

            # colour histogram
            masked_pixels        = hsv[:,:,:][img[:,:,3] == k]
            regions[k]["size"]   = len(masked_pixels / 4)
            regions[k]["hist_c"] = self.calc_colour_hist(masked_pixels)

            # texture histogram
            regions[k]["hist_t"] = self.calc_texture_hist(
                tex_grad[:,:][img[:,:,3]==k])

        return regions

    #--------------------------------------------------------------------------

    def extract_neighbours(self,regions):

        def intersect(a,b):
            if (a["min_x"] < b["min_x"] < a["max_x"]
                    and a["min_y"] < b["min_y"] < a["max_y"]) or (
                a["min_x"] < b["max_x"] < a["max_x"]
                    and a["min_y"] < b["max_y"] < a["max_y"]) or (
                a["min_x"] < b["min_x"] < a["max_x"]
                    and a["min_y"] < b["max_y"] < a["max_y"]) or (
                a["min_x"] < b["max_x"] < a["max_x"]
                    and a["min_y"] < b["min_y"] < a["max_y"]):
                return True
            return False

        R = list(regions.items())
        neighbours = []
        for cur, a in enumerate(R[:-1]):
            for b in R[cur + 1:]:
                if intersect(a[1], b[1]):
                    neighbours.append((a, b))

        return neighbours
