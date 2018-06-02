#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
AUVSI SUAS MISSION PLANNER MAIN
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
import numpy as np
import threading
import sys
import os

try:
    import auvsi_suas.python.src.mission_planner.interface as mission_planner
    import auvsi_suas.config as config
    
except ModuleNotFoundError as e:
    cwd = os.getcwd().split(os.path.sep)
    project_dir = os.path.sep.join(cwd[:cwd.index("auvsi_suas")])
    print('{}\n\nRun "export PYTHONPATH=$PYTHONPATH:{}"'.format(e,
                project_dir),file=sys.stderr)

"""
===============================================================================
MISSION PLANNER MAIN
===============================================================================
"""

class MissionPlannerMain(object):

    def __init__(self):
        self.interface = mission_planner.MPInterface()

    #--------------------------------------------------------------------------

    def main(self):
        self.interface.init_gs2mp_server()

"""
===============================================================================

===============================================================================
"""

def main():
    mp = MissionPlannerMain()
    mp.main()

if __name__ == "__main__":
    main()