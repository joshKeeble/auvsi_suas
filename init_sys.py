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
import subprocess
import importlib
import config
import site
import pip
import sys
import os
import re

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

ss_line = '-'*80
ds_line = '='*80

#------------------------------------------------------------------------------

def get_user_input(message):
    """Simple user input in the form of (y/n) or (0/1)"""
    user_input = str(input(message))
    if (user_input in ['yes','Yes','y','Y','1']):
        return 1
    else:
        return 0

#------------------------------------------------------------------------------

def cmd_line_sep(message):
    print("{}\n{}\n{}".format(ds_line,message,ds_line),file=sys.stderr)

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
        self.gospel = True
        self.env_ptrn = r'/(anaconda(\d*){}envs{}skynet)/'.format(
                            os.sep,os.sep)
        self.env_name = "skynet"

    #--------------------------------------------------------------------------

    def check_platform(self):
        """Check the sys platform"""
        print("{}\nChecking OS Platform...\n".format(ss_line),
            file=sys.stderr)
        #Switch statement, why not?
        linux_message = lambda : print("\n{}\n{}Use linux next time.\n".format(
            penguin," "*13),file=sys.stderr)

        good_job = lambda : print("\nRunning on linux, good job.\n",
            file=sys.stderr)

        platform_cases = {
            'linux' :   good_job,
            'win32' :   linux_message,
            'cygwin':   linux_message,
            'darwin':   linux_message
        }
        platform_cases[sys.platform]()

    #--------------------------------------------------------------------------

    def verify_py_version(self):
        """Check Python version"""
        print("{}\nChecking Python Verion...\n".format(ss_line),
            file=sys.stderr)
        [v0,v1] = (list(map(int,sys.version.split('|')[0].split('.')[:2])))
        if not (v0 > 2):
            raise Exception("Python 3.* is required, not Python 2")
        if (v1 < 6):
            print("Warning, System was tested with Python 3.6, \
                not Python 3.{}".format(v1),file=sys.stderr)
        else:
            print("Python version is correct (3.*)",file=sys.stderr)

    #--------------------------------------------------------------------------
    def verify_conda(self):
        """Check if Anaconda is installed"""
        print("{}\nChecking if Anaconda is installed...\n".format(ss_line),
            file=sys.stderr)
        if ('|' in sys.version):
            if not (sys.version.split('|')[1].split(' ')[0] in ['Anaconda,']):
                print("Warning, System was developed using Anaconda",
                    file=sys.stderr)
            else:
                print("Anaconda found",file=sys.stderr)

    #--------------------------------------------------------------------------

    def check_skynet(self,z):
        """Check if system path is run inside skynet environment"""
        if re.search(self.env_ptrn,z,flags=0):
            return 1
        return 0

    #--------------------------------------------------------------------------
    ########################################################################### FIX ENVIRONMENT ACTIVATION
    def activate_env(self):
        if (sys.platform in ["win32","cygwin"]):
            #os.system("activate {}".format(self.env_name))
            print("Activate the environment by 'activate {}'".format(
                self.env_name),file=sys.stderr)
        else:
            #os.system("source activate {}".format(self.env_name))
            print("Activate the environment by 'source activate {}'".format(
                self.env_name),file=sys.stderr)

    #--------------------------------------------------------------------------

    def check_conda_env(self):
        """Check to see if run inside of a conda environment"""
        print("{}\nChecking if run inside skynet env...\n".format(ss_line),
            file=sys.stderr)
        env = False
        for path in sys.path:
            if self.check_skynet(path):
                env = True
                print("Running inside skynet env",file=sys.stderr)
                break

        if not env:
            print("Not running inside skynet env, searching for skynet env.",
                file=sys.stderr)
            proc = subprocess.Popen(["conda info --envs"],
                stdout=subprocess.PIPE,shell=True)

            (out,err) = proc.communicate()
            out       = ' '.join(str(out).split())
            env_exits = False

            for z in (out.split('\\n')):
                if (self.env_name in (z.split(' '))):
                    env_exits = True

            if env_exits:
                self.activate_env()
            else:
                user_input = str(input("Create conda env? (y/n):\n"))
                if (user_input in ['yes','Yes','y','Y','1']):
                    os.system("conda create -n skynet python=3.6")
                else:
                    print("WARNING, You are running outside of an environment,\
                        be careful of conflicting package changes during \
                        installation",
                        file=sys.stderr)

    #--------------------------------------------------------------------------

    def sys_path_check(self): ################################################# FIX
        """Check if auvsi_suas is within system python path"""
        project_dir = os.path.dirname(os.getcwd())
        print(sys.path)
        if not (project_dir in sys.path):
            p = subprocess.Popen(["export PYTHONPATH=$PYTHONPATH:{}".format(
                project_dir)],stdout=subprocess.PIPE,shell=True)
            (out,err) = p.communicate()
            #export PYTHONPATH=$PYTHONPATH:/home/dev/python-files
            #site.addsitedir(project_dir)
            print('Run "export PYTHONPATH=$PYTHONPATH:{}"'.format(
                project_dir),file=sys.stderr)
        import auvsi_suas

    #--------------------------------------------------------------------------

    def main(self):
        self.verify_py_version()
        self.verify_conda()
        if self.gospel:
            self.check_platform()
        self.check_conda_env()
        self.sys_path_check()

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

    def check_module(self,z):
        """Check if module is importable"""
        if not isinstance(z,str):
            raise TypeError("Module input must be type str, not {}".format(
                type(z).__name__))
        if (z.count(' ')):
            raise Exception("Single word module args please")
        print(sys.modules)
        try:
            importlib.import_module(z,package=None)
            return 1
        except:
            return 0

    #--------------------------------------------------------------------------

    def install_module(self,z):
        """Install from str name of module"""
        for cmd in ["pip install {}".format(z),"conda install {}".format(z)]:
            os.system(cmd)
            if self.check_module(z):
                break

    #--------------------------------------------------------------------------


    def check_libs(self):
        """Check version of current libraries against required libraries"""
        print("{}\nChecking library...\n".format(ss_line),
            file=sys.stderr)
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
                    print("Module: {} not found, attempting install\n".format(
                        key),file=sys.stderr)
                    self.install_module(key)

                else:
                    print("Module: {} found\n".format(key),file=sys.stderr)

                ver = float(ver)
                if (key in module_info):
                    if (ver > float(module_info[key][:2])):
                        print("{} needs to be upgraded to {}".format(key,ver))

    #--------------------------------------------------------------------------

    def check_cv2(self):
        """See if cv2 can be imported"""
        try:
            import cv2
            return 1
        except ImportError:
            return 0

    #--------------------------------------------------------------------------

    def install_cv2(self):
        """Install OpenCV through multiple methods"""
        cmds = ["conda install -c menpo opencv","pip install opencv"]
        if (sys.platform == "linux"):
            cmds.append("sudo apt-get install opencv-python")
        for cmd in cmds:
            if not self.check_cv2():
                os.system(cmd)
            else:
                pass
        if not self.check_cv2():
            raise Exception("Opencv may have to be installed from source")

    #--------------------------------------------------------------------------

    def main(self):
        self.check_libs()
        if not self.check_cv2():
            self.install_cv2()


"""
===============================================================================
MAIN
===============================================================================
"""

welcome = """
===============================================================================
                                       ...
                                     ..''...
                                     ........
                                 ..  ........  .
                               ..'...  ....   ....
                             .........      ........
                           ..'.........    ...........
                         ..'............   .............
                       .................   ...............
                     ...................   .................
                    .,''''.....'....  .....'. ...'''''..,,,,'

                                SKYNET UAV SOFTWARE
                                --------------------

===============================================================================

System check for SKYNET Project, this will perform the following functions:

author = hal112358
-------------------------------------------------------------------------------

System Configuration Check:

    Function ----------------------------------------------- Required

    Check Python Version ----------------------------------- Yes
    Check Anaconda Installation ---------------------------- Yes
    Check if run inside Skynet conda env ------------------- No

-------------------------------------------------------------------------------
"""

def main():
    print(welcome)
    get_user_input("Continue? (y/n):\n")
    cmd_line_sep("SYSTEM CHECK")
    SYSCheck().main()
    cmd_line_sep("OPERATING SYSTEM CHECK")
    OSCheck().main()
    cmd_line_sep("LIBRARIES CHECK")
    if (get_user_input("Install required packages? (y/n):\n")):
        LibCheck().main()


if __name__ == "__main__":
    main()
