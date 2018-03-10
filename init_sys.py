#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
SYSTEM CONFIGURATION VALIDATION
===============================================================================
Check to see if the current system works with requirements.
-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
import importlib
import pip
import sys
import os

__author__ = "hal112358"

penguin = """
                 .88888888:.
                88888888.88888.
              .8888888888888888.
              888888888888888888
              88' _`88'_  `88888
              88 88 88 88  88888
              88_88_::_88_:88888
              88:::,::,:::::8888
              88`:::::::::'`8888
             .88  `::::'    8:88.
            8888            `8:888.
          .8888'             `888888.
         .8888:..  .::.  ...:'8888888:.
        .8888.'     :'     `'::`88:88888
       .8888        '         `.888:8888.
      888:8         .           888:88888
    .888:88        .:           888:88888:
    8888888.       ::           88:888888
    `.::.888.      ::          .88888888
   .::::::.888.    ::         :::`8888'.:.
  ::::::::::.888   '         .::::::::::::
  ::::::::::::.8    '      .:8::::::::::::.
 .::::::::::::::.        .:888:::::::::::::
 :::::::::::::::88:.__..:88888:::::::::::'
  `'.:::::::::::88888888888.88:::::::::'
        `':::_:' -- '' -'-' `':_::::'`

"""

"""
===============================================================================
SYS CHECK
===============================================================================
"""

class SYSCheck(object):
    """
    Check the system configurations
    """

    def __init__(self):
        self.gospel = False

    #--------------------------------------------------------------------------

    def check_platform(self):
        """Check the sys platform"""
        #Switch statement, why not?
        linux_message = lambda : print("\n{}\n\n{}Use linux next time\n".format(
            penguin," "*13),file=sys.stderr)

        platform_cases = {
            'linux' :   lambda : _,
            'win32' :   linux_message,
            'cygwin':   linux_message,
            'darwin':   linux_message
        }
        platform_cases[sys.platform]()

    #--------------------------------------------------------------------------

    def verify_py_version(self):
        """Check Python version"""
        [v0,v1] = (list(map(int,sys.version.split('|')[0].split('.')[:2])))
        if not (v0 > 2):
            raise Exception("Python 3.* is required, not Python 2")
        if (v1 < 6):
            print("Warning, System was tested with Python 3.6, \
                not Python 3.{}".format(v1),file=sys.stderr)

    #--------------------------------------------------------------------------
    def verify_conda(self):
        """Check if Anaconda is installed"""
        if not (sys.version.split('|')[1].split(' ')[0] in ['Anaconda,']):
            print("Warning, System was developed using Anaconda",
                file=sys.stderr)

    #--------------------------------------------------------------------------

    def main(self):
        self.verify_py_version()
        self.verify_conda()
        if self.gospel:
            self.check_platform()

"""
===============================================================================
OS CHECK
===============================================================================
"""

class OSCheck(object):
    """
    Check the os configurations
    """

    def __init__(self):
        pass

    #--------------------------------------------------------------------------

    def main(self):
        pass

"""
===============================================================================
LIBRARY CHECK
===============================================================================
"""

class LibCheck(object):
    """
    Check for missing packages
    """

    def __init__(self):
        pass

    #--------------------------------------------------------------------------

    def fetch_modules(self):
        """Return list of installed modules"""
        libs = pip.get_installed_distributions()
        return sorted(["{}=={}".format(i.key,i.version) for i in libs])

    #--------------------------------------------------------------------------

    def n_key_sep(self,z):
        """Seperate library key"""
        for n in ['==','<=','>=','<','>']:
            if n in z:
                return z.split(n)
        return [z]

    #--------------------------------------------------------------------------


    def check_libs(self):
        """Check version of current libraries against required libraries"""
        eq_sep      = lambda z: z.split('==')
        modules     = self.fetch_modules()
        module_info = list(map(eq_sep,modules))
        module_info = dict(module_info)

        requirement_path = "./requirements.txt"
        if not os.path.exists(requirement_path):
            raise Exception("Requirement file: {} not found\n".format(
                requirement_path))

        with open("./requirements.txt",'r') as lib_file:
            r_libs = lib_file.read().split('\n')
            r_libs.remove('')
            r_libs = list(map(self.n_key_sep,r_libs))

            for lib in r_libs:
                if (len(lib)-1):
                    (key,ver) = lib
                else:
                    key,ver = lib[0],0.

                if not (key in module_info):
                    raise Exception("Module: {} not found\n".format(key))

                ver = float(ver)
                if (ver > float(module_info[key][:2])):
                    print("{} needs to be upgraded to {}".format(key,ver))

    #--------------------------------------------------------------------------

    def main(self):
        self.check_libs()

"""
===============================================================================
MAIN
===============================================================================
"""

def main():
    SYSCheck().main()
    OSCheck().main()
    LibCheck().main()

if __name__ == "__main__":
    main()
