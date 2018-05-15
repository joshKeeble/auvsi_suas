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
from PIL import Image, ImageTk
from tkinter import ttk
import tkinter
import sys
import os


import auvsi_suas.python.src.communications.test_link as tcp_test
import auvsi_suas.config as config
 
class AUVSIUserInterface(ttk.Frame):
    """The adders gui and functions."""
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
        self.CONFIG_ROWSPAN = 1
        self.CONFIG_HEIGHT = 10
        self.CONFIG_COLUMNSPAN = 2
        self.CONFIG_WIDTH = 28
        self.CONFIG_STICK = 'ew'
        self.messages = {

            "interop"   :   {
                "c" :   "Interoperability System connected",
                "n" :   "Interoperability System not connected"
            },
            "mp"        :   {
                "c" :   "Mission Planner connected",
                "n" :   "Mission Planner not connected"
            },
            "pld"        :   {
                "c" :   "Payload connected",
                "n" :   "Payload not connected"
            },
            "oa"        :   {
                "d" :   "System deactivated",
                "n" :   "No virtual obstacles detected"
            },
            "trgt"      :   {
                "d" :   "System deactivated",
                "n" :   "No targets detected"
            },
            "gps"       :   {
                "n" :   "System deactivated"
            },
            "cls"       :   {
                "n" :   "No targets detected"
            }
        }
        self.init_gui()
    
    #--------------------------------------------------------------------------
 
    def on_quit(self):
        """Exits program."""
        quit()

    #--------------------------------------------------------------------------

    def mp_test_connection(self):
        """Test TCP connection to mission planner"""
        if tcp_test.tcp_connection_test(config.MISSION_PLANNER_HOST,
            config.MISSION_PLANNER_PORT):
            self.mp_answer_label['text'] = "Success, Mission Planner connected"
        else:
            self.mp_answer_label['text'] = "Failure, cannot connect to Mission Planner"

    #--------------------------------------------------------------------------

    def payload_test_connection(self):
        """Test TCP connection to payload"""
        if tcp_test.tcp_connection_test(config.MISSION_PLANNER_HOST,
            config.MISSION_PLANNER_PORT):
            self.pld_answer_label['text'] = "Success, Payload connected"
        else:
            self.pld_answer_label['text'] = "Failure, cannot connect to Payload"

    #--------------------------------------------------------------------------
 
    def calculate(self):
        """Calculates the sum of the two inputted numbers."""
        num1 = int(self.num1_entry.get())
        num2 = int(self.num2_entry.get())
        num3 = num1 + num2
        self.answer_label['text'] = num3

    #--------------------------------------------------------------------------
 
    def init_gui(self):
        """Builds GUI."""
        style = ttk.Style()
        style.configure("Output.TLabel",borderwidth=2, foreground="red")
        self.root.title('AUVSI SUAS User Interface')
        self.root.option_add('*tearOff', 'FALSE')
        self.root.geometry("1090x460")
 
        self.grid(column=0,row=0,sticky=self.CONFIG_STICK)
 
        self.menubar = tkinter.Menu(self.root)
 
        self.menu_file = tkinter.Menu(self.menubar)
        self.menu_file.add_command(label='Exit', command=self.on_quit)
 
        self.menu_edit = tkinter.Menu(self.menubar)
 
        self.menubar.add_cascade(menu=self.menu_file, label='File')
        self.menubar.add_cascade(menu=self.menu_edit, label='Edit')
 
        self.root.config(menu=self.menubar)
 
        # self.num1_entry = ttk.Entry(self, width=5)
        # self.num1_entry.grid(column=1, row = 2)
 
        # self.num2_entry = ttk.Entry(self, width=5)
        # self.num2_entry.grid(column=3, row=2)

        #----------------------------------------------------------------------
        # Interoperability System Login
        #----------------------------------------------------------------------
        current_column = 0
        current_row = 0
        # Labels that remain constant throughout execution.
        self.system_login_text = ttk.Label(self, text='Interoperability System Login')
        self.system_login_text.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,sticky=self.CONFIG_STICK)

        # Interopability System Separator
        current_row += 1
        self.interop_line = ttk.Separator(self,orient='horizontal')
        self.interop_line.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        # Interopability URL Entry
        current_row += 1
        self.url_text = ttk.Label(self,
            text='Interoperability URL',
            width=self.CONFIG_WIDTH
            )
        self.url_text.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )
        self.url_entry = ttk.Entry(self,
            width=self.CONFIG_WIDTH)
        self.url_entry.grid(
            column=current_column+1,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        # Interopability Username Entry Label
        current_row += 1
        self.username_text = ttk.Label(self,
            text='Interoperability Username',
            width=self.CONFIG_WIDTH
            )
        self.username_text.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        # Interopability Username Entry Label
        self.username_entry = ttk.Entry(self,
            width=self.CONFIG_WIDTH)
        self.username_entry.grid(
            column=current_column+1,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        # Interopability Password Entry Label
        current_row += 1
        self.password_text = ttk.Label(self,
            text='Interoperability Password',
            width=self.CONFIG_WIDTH
            )
        self.password_text.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        # Interopability Password Entry
        self.password_entry = ttk.Entry(self,
            width=self.CONFIG_WIDTH
            )
        self.password_entry.grid(
            column=current_column+1,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        # Submit Interopability Login Credentials
        current_row += 1
        self.calc_button = ttk.Button(self,
            text='Login',
            command=self.calculate,
            width=self.CONFIG_WIDTH*2
            )
        self.calc_button.grid(column=current_column,row=5,columnspan=self.CONFIG_COLUMNSPAN,rowspan=self.CONFIG_ROWSPAN,sticky=self.CONFIG_STICK)
 
        # Interopabi lity Login Notification Label
        current_row += 1
        self.answer_frame = ttk.LabelFrame(self,
            text='Status'
            )

        self.answer_frame.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )
 
        # Interopability Login Notification
        current_row += 1
        self.answer_label = ttk.Label(
            self.answer_frame,
            text=self.messages['interop']['n']
            )
        self.answer_label.configure(style="Output.TLabel")
        self.answer_label.grid(
            column=0,
            row=current_row
            )

        #----------------------------------------------------------------------
        # Test Connection to Mission Planner
        #----------------------------------------------------------------------
        current_column = 0
        current_row = 8
        # Mission Planner Test Label
        self.mp_test_text = ttk.Label(self,
            text='Mission Planner Connection Test',
            width=self.CONFIG_WIDTH
            )
        self.mp_test_text.grid(
            column=current_column,
            row=current_row,
            columnspan=1,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        # Mission Planner Separator
        current_row += 1
        self.mp_line = ttk.Separator(self,orient='horizontal')
        self.mp_line.grid(
            column=current_column,
            row=current_row,
            columnspan=1,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        # Mission Planner Test Connection Button
        current_row += 1
        self.calc_button = ttk.Button(self,
            text='Test Connection',
            command=self.mp_test_connection,
            width=self.CONFIG_WIDTH
            )
        self.calc_button.grid(
            column=current_column,
            row=current_row,
            columnspan=1,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )
 
        # Mission Planner Test Connection Label
        current_row += 1
        self.mp_answer_frame = ttk.LabelFrame(self,
            text='Status',
            # height=50,
            width=self.CONFIG_WIDTH
            )
        self.mp_answer_frame.grid(
            column=current_column,
            row=current_row,
            columnspan=1,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )
 
        # Mission Planner Test Connection Notification
        current_row += 1
        self.mp_answer_label = ttk.Label(
            self.mp_answer_frame,
            text=self.messages['mp']['n'],
            width=self.CONFIG_WIDTH
            )
        self.mp_answer_label.configure(style="Output.TLabel")
        self.mp_answer_label.grid(
            column=current_column,
            row=current_row,
            columnspan=1,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        #----------------------------------------------------------------------
        # Test Connection to Payload
        #----------------------------------------------------------------------
        current_column = 1
        current_row = 8
        # Payload Connection Test Label
        self.pld_test_text = ttk.Label(self,
            text='Payload Connection Test',
            width=self.CONFIG_WIDTH
            )
        self.pld_test_text.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        # Payload Connection Separator
        current_row += 1
        self.pld_line = ttk.Separator(self,orient='horizontal')
        self.pld_line.grid(
            column=current_column,
            row=current_row,
            columnspan=1,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        # Payload Connection Test Connection Button
        current_row += 1
        self.pld_button = ttk.Button(self,
            text='Test Connection',
            command=self.payload_test_connection,
            width=self.CONFIG_WIDTH
            )
        self.pld_button.grid(
            column=current_column,
            row=current_row,
            columnspan=1,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )
 
        # Payload Connection Test Connection Label
        current_row += 1
        self.pdl_answer_frame = ttk.LabelFrame(self,
            text='Status',
            # height=50,
            width=self.CONFIG_WIDTH
            )
        self.pdl_answer_frame.grid(
            column=current_column,
            row=current_row,
            columnspan=1,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )
 
        # Payload Connection Test Connection Notification
        current_row += 1
        self.pld_answer_label = ttk.Label(
            self.pdl_answer_frame,
            text=self.messages['pld']['n'],
            width=self.CONFIG_WIDTH
            )
        self.pld_answer_label.configure(style="Output.TLabel")
        self.pld_answer_label.grid(
            column=current_column,
            row=current_row,
            columnspan=1,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        #----------------------------------------------------------------------
        # Seek and Destroy
        #----------------------------------------------------------------------
        current_row = 0
        current_column = 8
        # 
        self.snd_label = ttk.Label(self,
            text='Targeting System')
        self.snd_label.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        # Seek and Destroy Separator
        current_row += 1
        self.snd_line = ttk.Separator(self,orient='horizontal')
        self.snd_line.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        # Seek and Destroy Emergent Target Standard Button
        current_row += 1
        self.snd_button = ttk.Button(self,
            text='Activate Standard Targeting',
            command=self.calculate,
            width=self.CONFIG_WIDTH
            )
        self.snd_button.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        # Seek and Destroy Emergent Target Activation Button
        current_row += 1
        self.snd_button = ttk.Button(self,
            text='Activate Emergent Targeting',
            command=self.calculate,
            width=self.CONFIG_WIDTH
            )
        self.snd_button.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )
 
        # Seek and Destroy Status
        current_row += 1
        self.snd_answer_frame = ttk.LabelFrame(self,
            text='Status',
            # height=50,
            width=self.CONFIG_WIDTH
            )
        self.snd_answer_frame.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )
 
        # Seek and Destroy Status Output
        current_row += 1
        self.snd_answer_label = ttk.Label(
            self.snd_answer_frame,
            text=self.messages['trgt']['n'],
            width=self.CONFIG_WIDTH
            )
        self.snd_answer_label.configure(style="Output.TLabel")
        self.snd_answer_label.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )
        
        # Seek and Destroy Classification
        current_row += 1
        self.snd_classification = ttk.LabelFrame(self,
            text='Classification',
            # height=50,
            width=self.CONFIG_WIDTH
            )
        self.snd_classification.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )
 
        # Seek and Destroy Classification Output
        current_row += 1
        self.snd_classification_label = ttk.Label(
            self.snd_classification,
            text='System deactivated',
            width=self.CONFIG_WIDTH
            )
        self.snd_classification_label.configure(style="Output.TLabel")
        self.snd_classification_label.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        # Seek and Destroy Location Label
        current_row += 1
        self.snd_location = ttk.LabelFrame(self,
            text='Location',
            # height=50,
            width=self.CONFIG_WIDTH
            )
        self.snd_location.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )
 
        # Seek and Destroy Location Output
        current_row += 1
        self.snd_location_label = ttk.Label(
            self.snd_location,
            text=self.messages['gps']['n'],
            width=self.CONFIG_WIDTH
            )
        self.snd_location_label.configure(style="Output.TLabel")
        self.snd_location_label.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        # Seek and Destroy Manual Mode Button
        current_row += 1
        self.snd_button = ttk.Button(self,
            text='Manual Mode',
            command=self.calculate,
            width=self.CONFIG_WIDTH
            )
        self.snd_button.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        # Seek and Destroy Review Button
        current_row += 1
        self.snd_button = ttk.Button(self,
            text='Review Objects',
            command=self.calculate,
            width=self.CONFIG_WIDTH
            )
        self.snd_button.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        # Seek and Destroy Upload Button
        current_row += 1
        self.upload_button = ttk.Button(self,
            text='Submit Objects',
            command=self.calculate,
            width=self.CONFIG_WIDTH
            )
        self.upload_button.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        #----------------------------------------------------------------------
        # Stealth
        #----------------------------------------------------------------------

        current_row = 0
        current_column = 4

        # Obstacle Avoidance Label
        self.oa_label = ttk.Label(self,
            text='Obstacle Avoidance System'
            )
        self.oa_label.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        # Obstacle Avoidance Separator
        current_row += 1
        self.oa_line = ttk.Separator(self,
            orient='horizontal'
            )
        self.oa_line.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        # Obstacle Avoidance Activation Button
        current_row += 1
        self.oa_button = ttk.Button(self,
            text='Activate',
            command=self.calculate
            )
        self.oa_button.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )
 
        # Obstacle Avoidance Status Label
        current_row += 1
        self.oa_answer_frame = ttk.LabelFrame(self,
            text='Status'
            #height=50
            )
        self.oa_answer_frame.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )
 
        # Obstacle Avoidance Status Notification
        current_row += 1
        self.oa_answer_label = ttk.Label(
            self.oa_answer_frame,
            text='No virtual obstacles detected'
            )
        self.oa_answer_label.configure(style="Output.TLabel")
        self.oa_answer_label.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        #----------------------------------------------------------------------
        # Payload Deployment
        #----------------------------------------------------------------------

        current_row = 8
        current_column = 4

        self.deploy_label = ttk.Label(self,
            text='Payload Deployment'
            )
        self.deploy_label.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        # Payload Deployment Separator
        current_row += 1
        self.deploy_label = ttk.Separator(self,
            orient='horizontal'
            )
        self.deploy_label.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )

        # Payload Deployment  Connection Button
        current_row += 1
        self.deploy_button = ttk.Button(self,
            text='Deploy',
            command=self.calculate
            )
        self.deploy_button.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )
 
        # Payload Deployment Status Label
        current_row += 1
        self.deploy_answer_frame = ttk.LabelFrame(self,
            text='Status'
            #height=50
            )
        self.deploy_answer_frame.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )
 
        # Payload Deployment Status Notification
        current_row += 1
        self.deploy_answer_label = ttk.Label(
            self.deploy_answer_frame, 
            text='System deactivated'
            )
        self.deploy_answer_label.configure(style="Output.TLabel")
        self.deploy_answer_label.grid(
            column=current_column,
            row=current_row,
            columnspan=self.CONFIG_COLUMNSPAN,
            rowspan=self.CONFIG_ROWSPAN,
            sticky=self.CONFIG_STICK
            )
        
        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=5)
 
if __name__ == '__main__':
    root = tkinter.Tk()
    AU(root)
    root.mainloop()