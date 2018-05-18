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
    import auvsi_suas.python.src.communications.osc_client = osc_client
    import auvsi_suas.python.src.communications.osc_server = osc_server
    
except ModuleNotFoundError as e:
    cwd = os.getcwd().split(os.path.sep)
    project_dir = os.path.sep.join(cwd[:cwd.index("auvsi_suas")])
    print('{}\n\nRun "export PYTHONPATH=$PYTHONPATH:{}"'.format(e,
                project_dir),file=sys.stderr)

"""
===============================================================================
Seek and Destory UAV Main
===============================================================================
"""

def handle(data): #------------------------------------------------------------ FIX
	print(data)


class SeekAndDestroyUAV(object):

	def __init__(self):
		self.payload_osc_group = config.PAYLOAD_OSC_GROUP
		self.payload_osc_port  = config.PAYLOAD_OSC_PORT
		self.gnd_osc_group     = config.GROUND_STATION_OSC_GROUP
		self.gnd_osc_port.     = config.GROUND_STATION_OSC_PORT

	#--------------------------------------------------------------------------

	def init_osc_client(self):
		"""Initialize OSC client to send data to ground station"""
		self.client = osc_client.OSCCLient(self.payload_osc_group,
			self.payload_osc_port)
		self.client.init_client()

	#--------------------------------------------------------------------------

	def init_osc_server(self):
		"""Initialize OSC server to recieve data from the ground station"""
		self.server = osc_server.OSCServer(self.gnd_osc_group,
			self.gnd_osc_port)
		self.server.init_server(handle)

	#--------------------------------------------------------------------------

	def main(self):
		self.init_osc_server()
		self.init_osc_client()



"""
===============================================================================
Seek and Destory Ground Station Main
===============================================================================
"""		

class SeekAndDestroyGroundStation(object):

	def __init__(self):
		self.osc_group = config.PAYLOAD_OSC_GROUP
		self.osc_port  = config.PAYLOAD_OSC_PORT

	#--------------------------------------------------------------------------
