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
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import numpy as np
import random
import shutil
import string
import cv2
import sys
import os

import auvsi_suas.config as config

__author__ = "hal112358"

"""
===============================================================================

===============================================================================
"""

class DataGeneration(object):

    def __init__(self):
        self.letter_x_shift = {
            'A' :   7,
            'B' :   8,
            'C' :   7,
            'D' :   7,
            'E' :   7,
            'F' :   7,
            'G' :   8,
            'H' :   9,
            'I' :   3,
            'J' :   7,
            'K' :   7,
            'L' :   6,
            'M' :   10,
            'N' :   9,
            'O' :   9,
            'P' :   9,
            'Q' :   9,
            'R' :   9,
            'S' :   8,
            'T' :   7,
            'U' :   8,
            'V' :   8,
            'W' :   9,
            'X' :   8,
            'Y' :   8,
            'Z' :   8
        }
        self.fonts = (
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_PLAIN,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
            cv2.FONT_ITALIC
        )
        self.shape_fncs = {
            "circle"        :   self.create_circle,
            "cross"         :   self.create_cross,
            "diamond"       :   self.create_diamond,
            "pentagon"      :   self.create_pentagon,
            "semicircle"    :   self.create_semicircle,
            "star"          :   self.create_star,
            "square"        :   self.create_square,
            "trapezoid"     :   self.create_trapezoid,
            "triangle"      :   self.create_triangle
        }
        self.ohev = lambda arg,target: 1. if (arg==target) else 0.
        self.letter_list = list(string.ascii_uppercase)

        self.init_system()
        self.load_backgrounds()

    #--------------------------------------------------------------------------

    def init_system(self):
        """Initialize the system file paths"""
        self.data_dir_path = "../../../../../generated_data"
        self.shape_data_dir = os.path.join(self.data_dir_path,"shapes")
        self.letter_data_dir = os.path.join(self.data_dir_path,"letters")
        if not os.path.exists(self.data_dir_path):
            os.mkdir(self.data_dir_path)

        for dir_path in [self.shape_data_dir,self.letter_data_dir]:

            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            else:
                if not len(os.listdir(dir_path)):
                    if config.CLEAN_GEN_DATA:
                        for file in os.listdir(dir_path):
                            target_file = os.path.join(dir_path,file)
                            if not (os.path.isdir(target_file)):
                                os.remove(target_file)

        shape_folders = list(map(lambda z: os.path.join(
            self.shape_data_dir,z),self.shape_fncs.keys()))

        letter_folders = list(map(lambda z: os.path.join(
            self.letter_data_dir,z),self.letter_list))

        for folder_list in [shape_folders,letter_folders]:
            for _dir in folder_list:
                if not os.path.exists(_dir):
                    os.mkdir(_dir)
                else:
                    if not os.path.isdir(_dir):
                        os.system("rm {}".format(_dir))
                    else:
                        if config.CLEAN_GEN_DATA:
                            shutil.rmtree(_dir)
                            os.mkdir(_dir)



    #--------------------------------------------------------------------------

    def load_backgrounds(self):
        """Load background images"""
        self.backgrounds = []
        background_images_dir = "./sample_images"
        if not os.path.exists(background_images_dir):
            raise Exception("Background image directory does not exist:{}".format(
                background_images_dir))
        for file in os.listdir(background_images_dir):
            full_path = os.path.join(background_images_dir,file)
            if full_path.endswith(".jpg"):
                self.backgrounds.append(cv2.imread(full_path))

    #--------------------------------------------------------------------------

    def add_noise(self,frame):
        """Add noise to a frame as to not overfit to background"""
        if config.DEBUG_MODE:
            if not isinstance(frame,np.ndarray):
                raise TypeError("Frame arg type not np.ndarray:{}".format(
                    type(frame).__name__))
        h,w,c = frame.shape
        gauss = np.random.uniform(0,random.randrange(1,50),(h,w,c))
        frame = frame + gauss.reshape(h,w,c)
        frame = np.clip(frame,0,255)
        frame = np.uint8(frame)
        return frame

    #--------------------------------------------------------------------------

    def gaussian_noise(self,frame):
        """Add Gaussian noise to a frame"""
        h,w,c = frame.shape
        gauss = np.random.uniform(self.gaussian_mean,self.gaussian_sigma,(h,w,c))
        frame = frame + gauss.reshape(h,w,c)
        frame = np.clip(frame,0,255)
        return frame


    #--------------------------------------------------------------------------

    def rotate1(self,frame,delta):
        """Rotate a frame by angle delta"""
        rows,cols = img.shape

        M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
        dst = cv2.warpAffine(img,M,(cols,rows))


    #--------------------------------------------------------------------------

    def resize(self,frame):
        """Random resize of frame"""
        h,w = frame.shape[:2]
        return cv2.resize(frame,((w+random.randrange(1,
            int(w/config.GEN_RESIZE_SCALE)),
            (h+int(h/config.GEN_RESIZE_SCALE)))))


    #--------------------------------------------------------------------------

    def rotate(self,image,angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    #--------------------------------------------------------------------------

    def create_triangle(self,frame,center,color):
        h,w     = frame.shape[:2]
        radius  = 35+random.randrange(-3,3)
        s       = int(radius/(2**0.5))
        shift   = 8

        (cw,ch) = center
        points  = [
            [cw-s,ch+s-shift],
            [cw+s,ch+s-shift],
            [cw,ch-s-shift]
            ]

        for i,p in enumerate(points):
            rw  = int(w/config.RNDN_SHAPE_RATIO)
            points[i][0] += random.randrange(-rw,rw)
            rh = int(h/config.RNDN_SHAPE_RATIO)
            points[i][1] += random.randrange(-rh,rh)

        vrx = np.asarray(points,dtype=np.int32)
        vrx = vrx.reshape((-1,1,2))
        return cv2.fillConvexPoly(frame,vrx,color)
        # return cv2.polylines(frame, [vrx], True, (0,255,255),3)

    #--------------------------------------------------------------------------

    def create_letter(self,frame,letter,center,color,r=25,font=cv2.FONT_HERSHEY_SIMPLEX):
        (cx,cy) = center
        cv2.putText(frame,letter,(cx-self.letter_x_shift[letter],cy),font,0.8,color, 2, cv2.LINE_AA)
        return frame

    #--------------------------------------------------------------------------

    def create_pentagon(self,frame,center,color):
        r  = 25+random.randrange(-3,3)
        # color = (0,0,255)
        (cw,ch) = center
        theta = 18
        alpha = 54

        points  = (
            (cw,ch-r-6), # 1
            (cw+int(r*np.cos(theta)-3),ch-int(r*np.sin(theta))-3), # 2
            (cw+int(r*np.cos(alpha)),ch+int(r*np.sin(alpha))), # 3
            (cw-int(r*np.cos(alpha)),ch+int(r*np.sin(alpha))), # 4
            (cw-int(r*np.cos(theta)-3),ch-int(r*np.sin(theta))-3)  # 5
            )
        frame = self.draw_points(frame,points,color)
        frame = self.draw_points(frame,points[1:],color)
        frame = self.draw_points(frame,points[2:],color)
        frame = self.draw_points(frame,points[3:],color)

        frame = self.draw_points(frame,points[0:3],color)
        frame = self.draw_points(frame,points[1:4],color)
        frame = self.draw_points(frame,(points[0],points[3],points[4]),color)

        return frame

    #--------------------------------------------------------------------------

    def create_trapezoid(self,frame,center,color):
        (cx,cy) = center
        short_side = 15
        long_side = 30
        height = 14
        y_shift = 5
        rshift = 3
        points = (
            (cx-short_side+random.randrange(-rshift,rshift),
            cy-height-y_shift+random.randrange(-rshift,rshift)),
            (cx-long_side+random.randrange(-rshift,rshift),
            cy+height-y_shift+random.randrange(-rshift,rshift)),
            (cx+short_side+random.randrange(-rshift,rshift),
            cy-height-y_shift+random.randrange(-rshift,rshift)),
            (cx+long_side+random.randrange(-rshift,rshift),
            cy+height-y_shift+random.randrange(-rshift,rshift))
            )
        frame = self.draw_points(frame,points,color)
        frame = self.draw_points(frame,points[:3],color)
        frame = self.draw_points(frame,points[1:],color)
        return frame


    #--------------------------------------------------------------------------

    def create_circle(self,frame,center,color):
        """Draw circle on frame"""
        r = 20+random.randrange(-3,3)
        (cx,cy) = center
        return cv2.circle(frame,(cx,cy-7),r,color,-1)

    #--------------------------------------------------------------------------

    def create_semicircle(self,frame,center,color):
        (cx,cy) = center
        return cv2.ellipse(frame,(cx,cy+8),(33,33),0,0,-180,color,-1)

    #--------------------------------------------------------------------------

    def create_star(self,frame,center,color):
        r       = 25+random.randrange(-3,3)
        # color   = (0,0,240)
        (cw,ch) = center
        theta   = 18
        alpha   = 54
        lr      = 1.5
        points  = (
            (cw,ch-int(lr*(r+4))), # 1
            (cw+int(lr*r*np.cos(theta)),ch-int(lr*r*np.sin(theta))), # 2
            (cw+int(lr*r*np.cos(alpha)),ch+int(lr*r*np.sin(alpha))), # 3
            (cw-int(lr*r*np.cos(alpha)),ch+int(lr*r*np.sin(alpha))), # 4
            (cw-int(lr*r*np.cos(theta)),ch-int(lr*r*np.sin(theta)))  # 5
            )
        print(points)
        frame = self.draw_points(frame,points[0:],color)
        cv2.circle(frame,(cw,int(ch-6)),int(lr*r/2.3),color,-1)
        return frame

    #--------------------------------------------------------------------------

    def create_diamond(self,frame,center,color):
        (cx,cy) = center
        short_side = 20
        long_side = 33
        y_shift = -5
        rshift = 3
        points = (
            (cx-short_side+random.randrange(-rshift,rshift),
            cy+y_shift+random.randrange(-rshift,rshift)),
            (cx+short_side+random.randrange(-rshift,rshift),
            cy+y_shift+random.randrange(-rshift,rshift)),
            (cx+random.randrange(-rshift,rshift),
            cy+y_shift-long_side+random.randrange(-rshift,rshift)),
            (cx+random.randrange(-rshift,rshift),
            cy+y_shift+long_side+random.randrange(-rshift,rshift))
            )
        frame = self.draw_points(frame,points,color)
        frame = self.draw_points(frame,points[:3],color)
        frame = self.draw_points(frame,points[1:],color)
        return frame


    #--------------------------------------------------------------------------

    def draw_points(self,frame,points,color):
        image = Image.fromarray(np.uint8(frame))
        draw = ImageDraw.Draw(image)
        draw.polygon((points), fill=color)
        frame = np.asarray(image)
        return frame

    #--------------------------------------------------------------------------

    def create_hexagon(self,frame,center):
        vrx = np.asarray([[20,80],[60,50],[100,80],[80,120],[40,120]],dtype=np.int32)
        vrx = vrx.reshape((-1,1,2))
        return cv2.polylines(frame, [vrx], True, (0,255,255),3)

    #--------------------------------------------------------------------------

    def create_square(self,frame,center,color):
        """Generate square shape"""
        margin = 18
        rmargin = 3
        (cx,cy) = center
        x1 = cx-(margin+random.randrange(-rmargin,rmargin))
        x2 = cx+(margin+random.randrange(-rmargin,rmargin))
        y1 = cy-(margin+random.randrange(-rmargin,rmargin))
        y2 = cy+(margin+random.randrange(-rmargin,rmargin))
        return cv2.rectangle(frame,(x1,y1),(x2,y2),color,-1)

    #--------------------------------------------------------------------------

    def create_cross(self,frame,center,color):
        """Generate cross shape"""
        short_side = 10
        long_side  = 30
        rmargin    = 3
        letter_shift = 6
        (cx,cy)    = center
        x1 = cx-short_side+random.randrange(-rmargin,rmargin)
        x2 = cx+short_side+random.randrange(-rmargin,rmargin)
        y1 = cy-long_side-letter_shift+random.randrange(-rmargin,rmargin)
        y2 = cy+long_side-letter_shift+random.randrange(-rmargin,rmargin)
        x3 = cx-long_side+random.randrange(-rmargin,rmargin)
        x4 = cx+long_side+random.randrange(-rmargin,rmargin)
        y3 = cx-short_side-letter_shift+random.randrange(-rmargin,rmargin)
        y4 = cy+short_side-letter_shift+random.randrange(-rmargin,rmargin)

        cv2.rectangle(frame,(x1,y1),(x2,y2),color,-1)
        return cv2.rectangle(frame,(x3,y3),(x4,y4),color,-1)


    #--------------------------------------------------------------------------

    def increase_brightness(self,frame):
        value = random.randrange(0,35)
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return frame



    #--------------------------------------------------------------------------

    def generate_data(self):
        frame_margin    = int(120*(2**0.5))
        subframe_margin = int(60*(2**0.5))
        n_background    = len(self.backgrounds)
        letters         = list(string.ascii_uppercase)
        count = 0
        for letter in letters:
            for n_shape_image in range(config.N_GENERATED_DATA):

                for shape in self.shape_fncs:
                    background_image = self.backgrounds[random.randrange(0,n_background)].copy()

                    background_image = np.uint8(background_image)

                    h,w = background_image.shape[:2]
                    (cx,cy) = (random.randrange(frame_margin,w-frame_margin),
                        random.randrange(frame_margin,h-frame_margin))


                    background_subframe = background_image[cy-subframe_margin:
                        cy+subframe_margin,cx-subframe_margin:cx+subframe_margin]

                    (cx,cy) = (subframe_margin,subframe_margin)

                    shape_color = (random.randrange(10,250),
                        random.randrange(10,250),random.randrange(10,250))

                    background_subframe = self.shape_fncs[shape](background_subframe,
                        (cx+random.randrange(-3,3),cy+random.randrange(-3,3)),
                        shape_color)

                    background_subframe = self.rotate(background_subframe,random.randrange(0,360))

                    letter_color = (random.randrange(10,250),
                        random.randrange(10,250),random.randrange(10,250))

                    letter_shift = 3

                    background_subframe = self.create_letter(
                        background_subframe,letter,
                        (cx+random.randrange(-letter_shift,letter_shift),
                        cy+random.randrange(-letter_shift,letter_shift)),
                        letter_color)

                    background_subframe = self.rotate(background_subframe,random.randrange(0,360))

                    rshift = 20

                    background_subframe = background_subframe[
                        cy-50+random.randrange(-rshift,rshift):
                        cy+50+random.randrange(-rshift,rshift),
                        cx-50+random.randrange(-rshift,rshift):
                        cx+50+random.randrange(-rshift,rshift)]

                    if random.randrange(0,5):
                        background_subframe = self.add_noise(background_subframe)

                    if random.randrange(0,2):
                        background_subframe = self.increase_brightness(background_subframe)
                    # background_subframe = self.desaturate(background_subframe)

                    del background_image

                    # letter_label = [self.ohev(n,letter) for n in self.letter_list]
                    # shape_label  = [self.ohev(n,shape) for n in self.shape_fncs]
                    # print(letter_label,shape_label)
                    filename = "{}_{}_{}.jpg".format(shape,letter,n_shape_image)

                    shape_dir = os.path.join(self.shape_data_dir,shape)
                    shape_path =  os.path.join(shape_dir,filename)

                    letter_dir = os.path.join(self.letter_data_dir,letter)
                    letter_path = os.path.join(letter_dir,filename)

                    background_subframe = cv2.resize(background_subframe,(100,100))

                    cv2.imwrite(letter_path,background_subframe)
                    cv2.imwrite(shape_path,background_subframe)



    #--------------------------------------------------------------------------

    def test(self):
        video_data = cv2.VideoCapture(0)
        center = (100,100)
        while True:
            _,frame = video_data.read()
            if random.randrange(0,2):
                frame = self.add_noise(frame)
            if random.randrange(0,2):
                frame = self.resize(frame)

            #cv2.imshow("frame1",frame)
            if random.randrange(0,2):
                frame = self.increase_brightness(frame)


            #frame = self.create_pentagon(frame,center)
            #frame = self.create_circle(frame,center,(255,255,255))

           #frame = self.create_semicircle(frame,center)

            frame = self.create_diamond(frame,center,(0,0,255))
            frame = self.create_letter(frame,'X',center,self.fonts[0])
            if isinstance(frame,np.ndarray):
                frame = self.add_noise(frame)

                cv2.imshow("frame2",frame)
                cv2.waitKey(1)

def main():
    dg = DataGeneration()
    dg.generate_data()

if __name__ == "__main__":
    main()
