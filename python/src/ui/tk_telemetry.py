#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
AUVSI SUAS User Interface
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk
from tkinter import ttk
import numpy as np
import threading
import requests
import unittest
import tkinter
import pickle
import time
import zlib
import sys
import os

import auvsi_suas.python.src.communications.osc_client as osc_client
import auvsi_suas.python.src.communications.osc_server as osc_server
import auvsi_suas.python.src.communications.test_link as tcp_test
import auvsi_suas.python.src.stealth.stealth_mode as stealth
import auvsi_suas.python.src.interop.client as client
import auvsi_suas.config as config


 
class TelemetryUI(ttk.Frame):
    """The adders gui and functions."""
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)

        self.root           = parent
        screen_width        = self.root.winfo_screenwidth()
        screen_height       = self.root.winfo_screenheight()
        self.width_ratio    = screen_width/2200
        self.height_ratio   = screen_height/1200


        self.root.geometry("{}x{}".format(
            int(800*self.width_ratio),int(460*self.height_ratio)))


        self.CONFIG_ROWSPAN     = 1
        self.CONFIG_HEIGHT      = int(10*self.height_ratio) ######################### FIX
        self.CONFIG_COLUMNSPAN  = 2
        self.CONFIG_WIDTH       = int(28*self.width_ratio)+23 ####################### FIX
        self.CONFIG_STICK       = 'ew'

        self.init_gui()
    

    def init_gui(self):
        """Builds GUI."""
        style = ttk.Style()
        style.configure("Output.TLabelFrame",borderwidth=2,size=50, foreground="black",bg='black',font=('Times',40))

        style.configure("Output.TLabel",borderwidth=2,size=50, foreground="black",bg='black',font=('Times',40))
        self.root.title('Telemetry')
        self.root.option_add('*tearOff', 'FALSE')
        
 
        self.grid(column=0,row=0,sticky=self.CONFIG_STICK)
 
        #----------------------------------------------------------------------
        # Interoperability System Login
        #----------------------------------------------------------------------
        current_column = 0
        current_row = 0

        # Interopabi lity Login Notification Label
        current_row += 1
        self.speed_label = ttk.LabelFrame(self,
            text='Speed'
            
            )
        self.speed_label.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        current_row += 1
        self.alt_label = ttk.LabelFrame(self,
            text='Alt'
            )

        self.alt_label.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        current_row += 1
        self.lat_label = ttk.LabelFrame(self,
            text='Latitude'
            )

        self.lat_label.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        current_row += 1
        self.long_label = ttk.LabelFrame(self,
            text='Longitude'
            )

        self.long_label.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )
 
        # Interopability Login Notification
        current_row += 1
        self.speed_text = ttk.Label(
            self.speed_label,
            text='Not connected',
            )
        self.speed_text.configure(style="Output.TLabel")
        self.speed_text.grid(
            column=0,
            row=current_row
            )

        # Interopability Login Notification
        current_row += 1
        self.alt_text = ttk.Label(
            self.alt_label,
            text='Not connected',
            )
        self.alt_text.configure(style="Output.TLabel")
        self.alt_text.grid(
            column=0,
            row=current_row
            )

        # Interopability Login Notification
        current_row += 1
        self.lat_text = ttk.Label(
            self.lat_label,
            text='Not connected',
            )
        self.lat_text.configure(style="Output.TLabel")
        self.lat_text.grid(
            column=0,
            row=current_row
            )

        # Interopability Login Notification
        current_row += 1
        self.long_text = ttk.Label(
            self.long_label,
            text='Not connected',
            )
        self.long_text.configure(style="Output.TLabel")
        self.long_text.grid(
            column=0,
            row=current_row
            )
        
        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=5)

    def update(self,speed,alt,lat,lng):
        speed_mode = 'meters'
        if speed_mode == 'meters':
            speed *= 1.943844
        self.speed_text.text = str(speed)
        self.speed_text.config(text=str(speed))

        self.alt_text.text = str(alt)
        self.alt_text.config(text=str(alt))

        self.lat_text.text = str(lat)
        self.lat_text.config(text=str(lat))

        self.long_text.text = str(lng)
        self.long_text.config(text=str(lng))
 
def init_telemtry_ui():
    global telemtry_ui
    root = tkinter.Tk()
    telemtry_ui = TelemetryUI(root)
    root.mainloop()


def activate_telemtry_ui():
    telemtry = threading.Thread(target=init_telemtry_ui,args=())
    telemtry.daemon = True
    telemtry.start()


if __name__ == '__main__':

    activate_telemtry_ui()
    time.sleep(1)
    while True:
        telemtry_ui.update(np.random.uniform(-10,10),
            np.random.uniform(-10,10),
            np.random.uniform(-10,10),
            np.random.uniform(-10,10)
            )
        time.sleep(1e-1)
