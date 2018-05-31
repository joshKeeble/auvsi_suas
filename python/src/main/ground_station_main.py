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

import tkinter as tk
import numpy as np
import threading
import sys
import os

try:
    import auvsi_suas.config as config
    import auvsi_suas.python.src.ui.menu.tk_menu as ui
    
except ModuleNotFoundError as e:
    cwd = os.getcwd().split(os.path.sep)
    project_dir = os.path.sep.join(cwd[:cwd.index("auvsi_suas")])
    print('{}\n\nRun "export PYTHONPATH=$PYTHONPATH:{}"'.format(e,
                project_dir),file=sys.stderr)


"""
===============================================================================
AUVSI SUAS MAIN CLASS FUNCTIONS
===============================================================================
"""

class AUVSIMain(object):

    def __init__(self):
        pass

    #--------------------------------------------------------------------------

    def init_ui(self):
        """Initialize User Interface"""
        root = tk.Tk()
        ui.AUVSIUserInterface(root)
        root.mainloop()

    #--------------------------------------------------------------------------

    def init_ui_thread(self):
        """Initialize User Interface Thread"""
        ui_thread = threading.Thread(target=self.init_ui,args=())
        ui_thread.daemon = True
        ui_thread.start()

    #--------------------------------------------------------------------------

    def main(self):
        self.init_ui()
        while True:
            continue

def main():
    auvsi = AUVSIMain()
    auvsi.main()

if __name__ == "__main__":
    main()

