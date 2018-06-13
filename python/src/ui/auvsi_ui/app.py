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
import numpy as np 
import sys
import os
import re


import auvsi_suas.python.src.communications.osc_client as osc_client
import auvsi_suas.python.src.communications.osc_server as osc_server
import auvsi_suas.python.src.communications.test_link as tcp_test
import auvsi_suas.python.src.deploy.activate as deployment
import auvsi_suas.python.src.stealth.stealth_mode as stealth
import auvsi_suas.python.src.interop.types as interop_types
import auvsi_suas.python.src.interop.client as client
import auvsi_suas.config as config




from flask import Flask, render_template, request, url_for, redirect
app = Flask(__name__,static_url_path='/static')

#@app.route('/')

#def login():
#   return render_template('login.html')
'''
@app.route('/')
@app.route('/login',methods=['POST'])
def login():
    #error = None
    print("Does this print?")
    if request.method == 'POST':
        print(request.form['username'])
        #if valid_login(request.form['username'],
        #              request.form['password']):
        #   return log_the_user_in(request.form['username'])
        #else:
        #    error = 'Invalid username/password'
    else:
        print("hmmm...")
        print(request.method)
        print(request.form)
        print(request.data)
        print(request.args)
        print(request)
        #print(request.form['username'])
    # the code below is executed if the request method
    # was GET or the credentials were invalid
    return render_template('login.html', error='error')
'''
#Make an app.route() decorator here
"""
@app.route('/', methods=['GET', 'POST']) #allow both GET and POST requests
def form_example():
    if request.method == 'POST':  #this block is only entered when the form is submitted
        language = request.form.get('language')
        framework = request.form['framework']

        return '''<h1>The language value is: {}</h1>
                  <h1>The framework value is: {}</h1>'''.format(language, framework)

    return '''<form method="POST">
                  Language: <input type="text" name="language"><br>
                  Framework: <input type="text" name="framework"><br>
                  <input type="submit" value="Submit"><br>
              </form>'''

#"""
"""
===============================================================================

===============================================================================
"""

class InteropParser(object):

    def __init__(self,client):
        self.interop_client = client

        self.drop_deploy = deployment.PayloadDeployment()

        self.stealth_mode = stealth.StealthMode()
        self.stealth_mode.set_min_altitude(150)
        self.stealth_mode.set_max_altitude(400)

    #--------------------------------------------------------------------------

    def payload_drop(self):
        self.drop_deploy.send_data()



    #--------------------------------------------------------------------------

    def run_stealth_mode(self):
        #global current_position ############################################## FIX
        i = 0
        while True:
            if self.stealth_mode_activated:
                obstacles = []
                for n in self.stealth_mode.obstacles:
                    try:
                        obstacles.append((n[0],n[1],n[2]))
                    except:
                        obstacles.append((n[0],n[1],n[2]))
                # print(obstacles)
                optimal_path = self.stealth_mode.find_path(obstacles,
                    self.stealth_mode.mission_waypoints,
                    self.stealth_mode.mission_waypoints[0],
                    self.stealth_mode.boundaries)
                try:


                    mp_client = mp_coms.MissionPlannerClient(host,port)

                    path = np.asarray(np.reshape(optimal_path,(1,-1))[0],dtype=np.float32).tolist()

                    print(path)

                    mp_client.send_data(path)

                    time.sleep(1)
                    mp_client.client_socket.close()

                except Exception as e:
                    exc_type,exc_obj,exc_tb = sys.exc_info()
                    fname = os.path.split(
                        exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type,fname,exc_tb.tb_lineno,e,
                        file=sys.stderr)
                i += 1
            else:
                time.sleep(1e-2)

    #--------------------------------------------------------------------------

    def fetch_mission(self):
        """Main function for processing interop data"""
        #self.init_telemetry_server()
        if config.INTEROP_USE_ASYNC:
            mission = self.interop_client.get_missions().result()[0]
        else:
            mission = self.interop_client.get_missions()[0]

        print(mission)
        self.drop_deploy.update_drop_zone(mission.air_drop_pos.latitude,
            mission.air_drop_pos.longitude)
        self.stealth_mode.set_home_location(mission.home_pos)
        self.stealth_mode.update_waypoints(mission.mission_waypoints)
        self.stealth_mode.set_boundaries(mission.fly_zones)

        # Create initial obstacles
        if config.INTEROP_USE_ASYNC:
            async_future = self.interop_client.get_obstacles()
            async_stationary, async_moving = async_future.result()
            self.all_obstacles = async_stationary+async_moving
        else:
            obstacle_list = self.interop_client.get_obstacles()
            stationary,moving = obstacle_list
            self.all_obstacles = stationary ################################## MOVING WAS REMOVED
        self.stealth_mode.update_obstacles(self.all_obstacles)


        # Set the current position of the vehicle
        # self.stealth_mode.set_current_position(current_position)

    #--------------------------------------------------------------------------

    def post_telemtry(self,lat,lng,alt,heading):
        t = interop_types.Telemetry(
            latitude=lat,
            longitude=lng, 
            altitude_msl=alt, 
            uas_heading=heading)

        if config.INTEROP_USE_ASYNC:
            mission = self.interop_client.post_telemetry(t).result()
        else:
            mission = self.interop_client.post_telemetry(t)


"""
===============================================================================

===============================================================================
"""


def server_handle(data,data_args):
    print(data)
    [lat,lng,alt,heading] = data
    interop_parser.post_telemetry(lat,lng,alt,heading)
    return b'1'

#------------------------------------------------------------------------------

def server_test():
    interface.init_autonomous_mode()
    mp_server = mp_coms.MissionPlannerServer(host,port)
    mp_server.listen(server_handle,0x00)

#------------------------------------------------------------------------------

def activate_telemetry_server():
    server_thread = threading.Thread(target=server_test,args=())
    server_thread.daemon = True
    server_thread.start()

#------------------------------------------------------------------------------

def login_get_function():
    return render_template('login.html', error='error')

#------------------------------------------------------------------------------

def login_post_function():
    global interop_client
    global interop_parser

    print(request.form)
    print(request.method)
    print(request.form['username'])
    print(request.data)
    print(request.args)
    interop_url = request.form['http_addr']
    username = request.form['username']
    password = request.form['password']
    timeout = 10
    if isinstance(username,str):
            if isinstance(password,str):
                if isinstance(interop_url,str):
                    try:
                        if config.INTEROP_USE_ASYNC:
                            interop_client = client.AsyncClient(
                                interop_url,username,password,timeout=timeout)
                        else:
                            interop_client = client.Client(
                                interop_url,username,password,timeout=timeout)
                        interop_parser = InteropParser(interop_client)
                        interop_parser.fetch_mission()
                        activate_telemetry_server()
                        return redirect(url_for('index'))
                    except Exception as e:
                        print(e,file=sys.stderr)
                        return redirect(url_for('access_denied'))
    return redirect(url_for('access_denied'))


@app.route("/", methods = ['GET', 'POST'])
def login():
    if request.method == 'GET':
        return login_get_function()

    elif request.method == 'POST':
        return login_post_function()

"""
===============================================================================

===============================================================================
"""

@app.route("/index", methods = ['GET','POST'])
def index():
    return render_template('index.html', error='error')

"""
===============================================================================

===============================================================================
"""

@app.route("/access_denied", methods = ['GET','POST'])
def access_denied():
    return render_template('access_denied.html', error='error')

"""
===============================================================================

===============================================================================
"""

@app.route("/documentation", methods = ['GET','POST'])
def documentation():
    return render_template('documentation.html', error='error')

"""
===============================================================================

===============================================================================
"""

def verify_ip(host):
    """Boolean verification of acceptable ip address"""
    valid = False
    if not (re.match(r'^((\d){1,3}.){3}(\d{1,3})$',host,re.M|re.I)):
        print("Invalid host argument:{}".format(host),file=sys.stderr)
    else:
        valid = True
    return valid

#------------------------------------------------------------------------------

def verify_port(port):
    """Boolean verification of acceptable port number"""
    if not isinstance(port,int):
        print("Invalid port type:{}".format(type(port).__name__),
            file=sys.stderr)
        return False
    if not (port >= 0 and port < 65535):
        print("Invalid port range (0-65535): {}".format(port),file=sys.stderr)
        return False
    return True

#------------------------------------------------------------------------------

def network_configuration_get_function():
    return render_template('network_configuration.html',error='error')

#------------------------------------------------------------------------------

def network_configuration_post_function():
    print(request.form)
    ground_station_addr = request.form['ground_station_addr']
    mission_planner_addr = request.form['mission_planner_addr']
    payload_addr = request.form['payload_addr']
    valid = False
    try:
        gs2mp_port = int(request.form['gs2mp_port'])
        valid = True
    except Exception as e:
        print("Incorrect gs2mp_port arg type, cannot convert to int:{}".format(
            e),file=sys.stderr)

    try:
        gs2pd_port = int(request.form['gs2pd_port'])
        valid = True
    except Exception as e:
        print("Incorrect gs2pd_port arg type, cannot convert to int:{}".format(
            e),file=sys.stderr)
        valid = False

    if valid:
        print(ground_station_addr)
        if verify_ip(ground_station_addr):
            print("changed")
            config.GROUND_STATION_HOST = ground_station_addr
        if verify_ip(mission_planner_addr):
            print("changed")
            config.MISSION_PLANNER_HOST = mission_planner_addr
        if verify_ip(payload_addr):
            print("changed")
            config.PAYLOAD_HOST = payload_addr

        if verify_port(gs2mp_port):
            print("changed")
            config.GROUND_STATION2MISSION_PLANNER_PORT = gs2mp_port
        if verify_port(gs2pd_port):
            print("changed")
            config.GROUND_STATION2PAYLOAD_PORT = gs2pd_port
        print(config.GROUND_STATION_HOST)
        print(config.MISSION_PLANNER_HOST)
        print(config.PAYLOAD_HOST)
        print(config.GROUND_STATION2MISSION_PLANNER_PORT)
        print(config.GROUND_STATION2PAYLOAD_PORT)
    return redirect(url_for('index'))

#------------------------------------------------------------------------------

@app.route("/network_configuration",methods=['GET','POST'])
def network_configuration():
    if request.method == 'GET':
        return network_configuration_get_function()
    elif request.method == 'POST':
        return network_configuration_post_function()
"""
===============================================================================

===============================================================================
"""


def review_objects_get_function():
    return render_template('review_objects_wizard.html',error='error')

def review_objects_post_function():
    print(request.form)
    print(request.method)
    print(request.data)
    print(request.args)
    return redirect(url_for('index'))


@app.route("/object_review",methods=['GET','POST'])
def review_objects():
    if request.method == 'GET':
        return review_objects_get_function()
    elif request.method == 'POST':
        return review_objects_post_function()

"""
===============================================================================

===============================================================================
"""

@app.route("/under_construction",methods=['GET','POST'])
def under_construction():
    return render_template('under_construction.html',error='error')

"""
===============================================================================

===============================================================================
"""

def payload_deployment_get_function():
    return render_template('payload_deployment.html',error='error')

def payload_deployment_post_function():
    print("DEPLOYED!!!!")
    return redirect(url_for('index'))


@app.route("/payload_deployment",methods=['GET','POST'])
def payload_deployment():
    if request.method == 'GET':
        return payload_deployment_get_function()
    elif request.method == 'POST':
        return payload_deployment_post_function()

"""
===============================================================================

===============================================================================
"""

def stationary_obstacle_avoidance_get_function():
    return render_template('obstacle_avoidance_stationary.html',error='error')

def stationary_obstacle_avoidance_post_function():
    print(request.form)
    print(request.method)
    print(request.data)
    print(request.args)
    return redirect(url_for('index'))


@app.route("/stationary_obstacle_avoidance",methods=['GET','POST'])
def stationary_obstacle_avoidancet():
    if request.method == 'GET':
        return stationary_obstacle_avoidance_get_function()
    elif request.method == 'POST':
        return stationary_obstacle_avoidance_post_function()


"""
===============================================================================

===============================================================================
"""

def load_manual_missions_get_function():
    return render_template('load_mission_wizard.html',error='error')

def load_manual_missions_post_function():
    print(request.form)
    print(request.method)
    print(request.data)
    print(request.args)
    return redirect(url_for('index'))


@app.route("/load_manual_missions",methods=['GET','POST'])
def load_manual_missions():
    if request.method == 'GET':
        return load_manual_missions_get_function()
    elif request.method == 'POST':
        return load_manual_missions_post_function()

"""
===============================================================================

===============================================================================
"""

def manual_targeting_missions_get_function():
    return render_template('manual_targeting.html',error='error')

def manual_targeting_missions_post_function():
    print("manual_targeting")
    print(request.form)
    print(request.method)
    print(request.data)
    print(request.args)
    return redirect(url_for('index'))


@app.route("/manual_targeting",methods=['GET','POST'])
def manual_targeting():
    if request.method == 'GET':
        return manual_targeting_missions_get_function()
    elif request.method == 'POST':
        return manual_targeting_missions_post_function()

"""
===============================================================================

===============================================================================
"""

def submit_objects_get_function():
    return render_template('submit_objects.html',error='error')

def submit_objects_post_function():
    print(request.form)
    print(request.method)
    print(request.data)
    print(request.args)
    return redirect(url_for('index'))


@app.route("/submit_objects",methods=['GET','POST'])
def submit_objects():
    if request.method == 'GET':
        return submit_objects_get_function()
    elif request.method == 'POST':
        return submit_objects_post_function()

"""
===============================================================================

===============================================================================
"""

if __name__ == '__main__':
   app.run(debug = True)