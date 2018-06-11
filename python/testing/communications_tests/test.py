import sys
import os
from os.path import expanduser
home = expanduser("~")
print(home)
cwd = os.getcwd()
cwd_split = cwd.split(os.sep)
sys.path.append(os.sep.join(cwd_split[:cwd_split.index('auvsi_suas')]))
#sys.path.append()

import auvsi_suas.python.src.communications.mp_communications as mp_coms

def server_handle(data,data_args):
	print(data)
	print(type(data))
	return b'1'

host = 'localhost'
port = 4011

def server_test():
	try:
		mp_server = mp_coms.MissionPlannerServer(host,port)
		mp_server.listen(server_handle,0x00)
	except Exception as e:
		print(e)

def client_test():

	mp_client = mp_coms.MissionPlannerClient(host,port)
	while True:
		try:
			data = [1,2,3,4,5]

			mp_client.send_data(data)
			print('Data sent:{}'.format(data))
			time.sleep(0.5)
		except Exception as e:
			print(e)
			print("Are you sure that the server is turned on?")

def main():
	if sys.argv[1] == 'c':
		client_test()
	elif sys.argv[1] == 's':
		server_test()
	else:
		raise Exception("Incorrect sys arg [s or c]")

main()