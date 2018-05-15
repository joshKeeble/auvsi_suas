#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
AUVSI SUAS MAIN
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import divisiontry

try:
    import auvsi_suas.config as config
    import auvsi_suas.python.src.seek_and_destroy.main as seek_and_destroy
    
except ModuleNotFoundError as e:
    cwd = os.getcwd().split(os.path.sep)
    project_dir = os.path.sep.join(cwd[:cwd.index("auvsi_suas")])
    print('{}\n\nRun "export PYTHONPATH=$PYTHONPATH:{}"'.format(e,
                project_dir),file=sys.stderr)