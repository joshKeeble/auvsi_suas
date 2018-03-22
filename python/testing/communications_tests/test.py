from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys

import tcp_test

def import_fix():
    id        = "auvsi_suas"
    path      = os.getcwd()
    path_list = path.split(os.path.sep)
    dir_path  = os.path.sep.join(path_list[:path_list.index(id)+1])
    sys.path.append(dir_path)

import_fix()

print(sys.path)

import auvsi_suas.python.testing.test2
