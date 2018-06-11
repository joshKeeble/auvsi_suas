import sys

paths = ['', 'C:\\Users\\soffer\\Desktop',
'C:\\Users\\soffer\\AppData\\Local\\Continuum\\anaconda3\\envs\\mission_planner\\python27.zip', 
'C:\\Users\\soffer\\AppData\\Local\\Continuum\\anaconda3\\envs\\mission_planner\\DLLs', 
'C:\\Users\\soffer\\AppData\\Local\\Continuum\\anaconda3\\envs\\mission_planner\\lib', 
'C:\\Users\\soffer\\AppData\\Local\\Continuum\\anaconda3\\envs\\mission_planner\\lib\\plat-win', 
'C:\\Users\\soffer\\AppData\\Local\\Continuum\\anaconda3\\envs\\mission_planner\\lib\\lib-tk', 
'C:\\Users\\soffer\\AppData\\Local\\Continuum\\anaconda3\\envs\\mission_planner', 
'C:\\Users\\soffer\\AppData\\Local\\Continuum\\anaconda3\\envs\\mission_planner\\lib\\site-packages']

for path in paths:
    sys.path.append(path)

import auvsi_suas.python.src.communications.mp_communications as mp_coms
import time

try:
    import clr
    clr.AddReference("MissionPlanner")
    import MissionPlanner
    clr.AddReference("MAVLink")
    import MAVLink
    clr.AddReference("MissionPlanner.Utilities")
except:
    print("ERROR: Not run within Mission Planner")



class MPInterface(object):

    def __init__(self):
        pass

    #--------------------------------------------------------------------------

    def signal_servo(self,servo=None,pwm=None):
        """
        The servo is servo 9, pwm is 1200 for open/drop, pwm is 1500 for closed.
        """
        MAV.doCommand(MAVLink.MAV_CMD.DO_SET_SERVO,servo,pwm,0,0,0,0,0)

    #--------------------------------------------------------------------------

    def open_payload(self):
        """Open the payload claw"""
        self.signal_servo(servo=9,pwm=1200)

    #--------------------------------------------------------------------------

    def close_payload(self):
        """Close the payload claw"""
        self.signal_servo(servo=9,pwm=1500)

    #--------------------------------------------------------------------------

    def init_autonomous_mode(self):
        if not (self.fetch_mode() == "Guided"):
            Script.ChangeMode("Guided")

    #--------------------------------------------------------------------------

    def init_gs2mp_server(self):
        """Initialize ground station to mission planner server"""
        self.gs2mp_server = osc_server.OSCServer(config.GROUND_STATION_HOST,
            config.GROUND_STATION2MISSION_PLANNER_PORT)
        self.gs2mp_server.init_server(gs2mp_server_handle)
        self.gs2mp_server.activate_listen_thread()

    #--------------------------------------------------------------------------

    def add_waypoint(self,lat,lng,alt):
        """Add a waypoint"""
        wp = MissionPlanner.Utilities.Locationwp()
        MissionPlanner.Utilities.Locationwp.lat.SetValue(wp,lat)
        MissionPlanner.Utilities.Locationwp.lng.SetValue(wp,lng)
        MissionPlanner.Utilities.Locationwp.alt.SetValue(wp,alt)
        MAV.setGuidedModeWP(wp)

    #--------------------------------------------------------------------------

    def fetch_mode(self):
        """Return the flying mode as string"""
        return cs.mode


host = 'localhost'
port = 4023

interface = MPInterface()

def server_handle(data,data_args):
    print(data)
    print(type(data))
    for i in range(len(data)/3):
        lat = data[3*i]
        lng = data[3*i+1]
        alt = data[3*i+2]
        interface.add_waypoint(lat,lng,alt)
    return b'1'


def server_test():
    interface.init_autonomous_mode()
    mp_server = mp_coms.MissionPlannerServer(host,port)
    mp_server.listen(server_handle,0x00)


def client_test():

    mp_client = mp_coms.MissionPlannerClient(host,port)

    while True:
        try:
            data = [1,2,3,4,5]

            mp_client.send_data(data)
            print('Data sent:{}'.format(data))
        except Exception as e:
            print(e)
            print("Are you sure that the server is turned on?")

server_test()