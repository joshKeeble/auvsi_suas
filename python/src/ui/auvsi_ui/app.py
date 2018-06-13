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
import threading
import random
import json
import sys
import os
import re

import auvsi_suas.python.src.communications.mp_communications as mp_coms
import auvsi_suas.python.src.communications.osc_client as osc_client
import auvsi_suas.python.src.communications.osc_server as osc_server
import auvsi_suas.python.src.communications.test_link as tcp_test
import auvsi_suas.python.src.stealth.stealth_mode as stealth
import auvsi_suas.python.src.interop.types as interop_types
import auvsi_suas.python.src.deploy.activate as deployment
import auvsi_suas.python.src.interop.client as client
import auvsi_suas.config as config


from flask import Flask, render_template, request, url_for, redirect

app = Flask(__name__,static_url_path='/static')

"""
===============================================================================

===============================================================================
"""

v_alt = 0

class InteropParser(object):

    def __init__(self,client):
        self.interop_client = client

        self.drop_deploy = deployment.PayloadDeployment()

        self.stealth_mode = stealth.StealthMode()
        self.stealth_mode.set_min_altitude(150)
        self.stealth_mode.set_max_altitude(400)

    #--------------------------------------------------------------------------

    def run_stealth_mode(self):
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
            mp_client = mp_coms.MissionPlannerClient(
                config.MISSION_PLANNER_HOST,
                config.GROUND_STATION2MISSION_PLANNER_PORT)

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
            self.interop_client.post_telemetry(t).result()
        else:
            self.interop_client.post_telemetry(t)

"""
===============================================================================

===============================================================================
"""


def server_handle(data,data_args):
    print(data)
    global v_lat,v_lng,v_alt,v_heading
    [v_lat,v_lng,v_alt,v_heading] = data
    interop_parser.post_telemetry(v_lat,v_lng,v_alt,v_heading)
    return b'1'

#------------------------------------------------------------------------------

def server_test():
    # interface.init_autonomous_mode()
    mp2gs_server = mp_coms.MissionPlannerServer(config.GROUND_STATION_HOST,
        config.MISSION_PLANNER2GROUND_STATION_PORT)
    mp2gs_server.listen(server_handle,0x00)

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
        if verify_ip(ground_station_addr):
            config.GROUND_STATION_HOST = ground_station_addr

        if verify_ip(mission_planner_addr):
            config.MISSION_PLANNER_HOST = mission_planner_addr

        if verify_ip(payload_addr):
            config.PAYLOAD_HOST = payload_addr

        if verify_port(gs2mp_port):
            config.GROUND_STATION2MISSION_PLANNER_PORT = gs2mp_port

        if verify_port(gs2pd_port):
            config.GROUND_STATION2PAYLOAD_PORT = gs2pd_port

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
    global v_alt
    try:
        drop_lat = interop_parser.drop_deploy.drop_zone_lat
        drop_lng = interop_parser.drop_deploy.drop_zone_lng
    except:
        drop_lat = 45.632676
        drop_lng = -122.651599
    if not v_alt:
        drop_height = 400
    else:
        drop_height = v_alt
    if (request.form['latitude'] != '' and request.form['longitude'] != ''):
        try:
            temp_drop_lat = float(request.form['latitude'])
            temp_drop_lng = float(request.form['longitude'])
            drop_lat,drop_lng = temp_drop_lat,temp_drop_lng
        except Exception as e:
            print("Error processing new drop lat/lng args:{}".format(e),
                file=sys.stderr)
    try:

        mp_client = mp_coms.MissionPlannerClient(
            config.MISSION_PLANNER_HOST,
            config.GROUND_STATION2MISSION_PLANNER_PORT)

        path = [(0,0,0),(drop_lat,drop_lng,drop_height)]

        path = np.asarray(np.reshape(path,(1,-1))[0],dtype=np.float32).tolist()

        mp_client.send_data(path)

        mp_client.client_socket.close()

    except Exception as e:
        exc_type,exc_obj,exc_tb = sys.exc_info()
        fname = os.path.split(
            exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type,fname,exc_tb.tb_lineno,e,
            file=sys.stderr)



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

def return_home_get_function():
    return render_template('return_home.html',error='error')

#------------------------------------------------------------------------------

def return_home_post_function():
    global v_alt
    try:
        home_lat = interop_parser.drop_deploy.drop_zone_lat
        home_lng = interop_parser.drop_deploy.drop_zone_lng
    except:
        home_lat = 45.632676
        home_lng = -122.651599
    home_height = 0
    if (request.form['latitude'] != '' and request.form['longitude'] != ''):
        try:
            temp_home_lat = float(request.form['latitude'])
            temp_home_lng = float(request.form['longitude'])
            home_lat,drop_lng = temp_home_lat,temp_home_lng
        except Exception as e:
            print("Error processing new drop lat/lng args:{}".format(e),
                file=sys.stderr)
    try:

        mp_client = mp_coms.MissionPlannerClient(
            config.MISSION_PLANNER_HOST,
            config.GROUND_STATION2MISSION_PLANNER_PORT)

        path = [(-1,-1,-1),(drop_lat,drop_lng,0)]

        path = np.asarray(np.reshape(path,(1,-1))[0],dtype=np.float32).tolist()

        mp_client.send_data(path)

        mp_client.client_socket.close()

    except Exception as e:
        exc_type,exc_obj,exc_tb = sys.exc_info()
        fname = os.path.split(
            exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type,fname,exc_tb.tb_lineno,e,
            file=sys.stderr)

    return redirect(url_for('index'))

#------------------------------------------------------------------------------

@app.route("/return_home",methods=['GET','POST'])
def return_home():
    if request.method == 'GET':
        return return_home_get_function()
    elif request.method == 'POST':
        return return_home_post_function()

"""
===============================================================================

===============================================================================
"""

def new_waypoint_get_function():
    return render_template('new_waypoint.html',error='error')

#------------------------------------------------------------------------------

def new_waypoint_post_function():
    if (request.form['latitude'] != '' 
        and request.form['longitude'] != '' 
        and request.form['altitude'] != ''):
        try:
            temp_new_lat = float(request.form['latitude'])
            temp_new_lng = float(request.form['longitude'])
            temp_new_alt = float(request.form['altitude'])
            new_lat,new_lng,new_alt = temp_home_lat,temp_home_lng,temp_new_alt
            try:
                mp_client = mp_coms.MissionPlannerClient(
                    config.MISSION_PLANNER_HOST,
                    config.GROUND_STATION2MISSION_PLANNER_PORT)

                path = [(-1,-1,-1),(new_lat,new_lng,new_alt)]

                path = np.asarray(np.reshape(path,(1,-1))[0],dtype=np.float32).tolist()

                mp_client.send_data(path)

                mp_client.client_socket.close()

            except Exception as e:
                exc_type,exc_obj,exc_tb = sys.exc_info()
                fname = os.path.split(
                    exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type,fname,exc_tb.tb_lineno,e,
                    file=sys.stderr)
        except Exception as e:
            print("Error processing new drop lat/lng args:{}".format(e),
                file=sys.stderr)

    return redirect(url_for('index'))

#------------------------------------------------------------------------------

@app.route("/new_waypoint",methods=['GET','POST'])
def new_waypoint():
    if request.method == 'GET':
        return new_waypoint_get_function()
    elif request.method == 'POST':
        return new_waypoint_post_function()

"""
===============================================================================

===============================================================================
"""

def stationary_obstacle_avoidance_get_function():
    return render_template('obstacle_avoidance_stationary.html',error='error')

def stationary_obstacle_avoidance_post_function():
    print("OBSTACLES!!!")
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

def save_object(target_type,shape,shape_color,letter,letter_color):
    cwd = os.getcwd().split(os.path.sep)
    project_dir = os.path.sep.join(cwd[:cwd.index("auvsi_suas")+1])
    object_dir = os.path.join(project_dir,'objects')

    n_objects = 0
    for n in os.listdir(object_dir):
        if n.endswith('.json'):
            n_objects += 1

    file_name = os.path.join(object_dir,'{}.json'.format(n_objects+1))

    target_type = target_type.lower()

    if (target_type in ['standard','emergent']):
        object_data = {
        "type": "{}".format(target_type),
        "latitude": 38.1478+np.random.uniform(-0.001,0.001),
        "longitude": -76.4275+np.random.uniform(-0.001,0.001),
        "orientation": ["n","w","e","s"][random.randrange(0,3)],
        "shape":"{}".format(shape),
        "background_color":"{}".format(shape_color),
        "alphanumeric":"{}".format(letter),
        "alphanumeric_color":"{}".format(letter_color)
        }
        print(file_name)
        with open(file_name,'w') as outfile:
            json.dump(object_data,outfile)

def manual_targeting_missions_get_function():
    return render_template('manual_targeting.html',error='error')

#------------------------------------------------------------------------------


def manual_targeting_missions_post_function():
    target_type = request.form['target_type']
    shape = request.form['shape_type']
    shape_color = request.form['shape_color']
    letter = request.form['alphanumeric']
    letter_color = request.form['alpha_color']
    save_object(target_type,shape,shape_color,letter,letter_color)
    return redirect(url_for('index'))

#------------------------------------------------------------------------------

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



def autonomous_targeting_missions_get_function():
    return render_template('autonomous_targeting.html',error='error')

def autonomous_targeting_missions_post_function():
    print("autonomous_targeting")
    print(request.form)
    print(request.method)
    print(request.data)
    print(request.args)
    return redirect(url_for('index'))


@app.route("/autonomous_targeting",methods=['GET','POST'])
def autonomous_targeting():
    if request.method == 'GET':
        return autonomous_targeting_missions_get_function()
    elif request.method == 'POST':
        return autonomous_targeting_missions_post_function()

"""
===============================================================================

===============================================================================
"""

def interop_submit_objects():
    dir_path = os.getcwd().split(os.sep)
    dir_path = os.sep.join(dir_path[dir_path.index('auvsi_suas')])


def submit_objects_get_function():
    return render_template('submit_objects.html',error='error')

def submit_objects_post_function():
    interop_submit_objects()
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