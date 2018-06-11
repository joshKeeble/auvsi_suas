import threading
import pickle
import socket
import struct
import time
import os

VIDEO_STREAM_RECV_SIZE = 1024

"""
===============================================================================
Client Object
===============================================================================
"""

class MissionPlannerClient(object):

    def __init__(self,host,port):
        self.host = host
        self.port = port
        self.setup_client()

    #--------------------------------------------------------------------------

    def setup_client(self):
        self.client_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.client_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        #print("We got here!")
        self.client_socket.connect((self.host,self.port))
        print("Client connected to {}:{}".format(self.host,self.port))

    #--------------------------------------------------------------------------

    def send_data(self,data):

        print(data)
        print('-'*80)
        data = pickle.dumps(data,protocol=2)

        print(data)
        print('-'*80)

        #data = struct.pack("L",len(data))+data

        print(data)
        print('-'*80)

        self.client_socket.sendall(data)

        print("Data sent, waiting for response")

        #server_response = self.client_socket.recv(10)

        #print(server_response)



"""
===============================================================================
Server Object
===============================================================================
"""

class MissionPlannerServer(object):

    def __init__(self,host,port):
        self.host = host
        self.port = port
        self.setup_server()

    #--------------------------------------------------------------------------

    def setup_server(self):
        self.server_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        self.server_socket.bind((self.host,self.port))
        print("Server activated on {}:{}".format(self.host,self.port))

    #--------------------------------------------------------------------------

    def listen(self,handle,handle_args):
        print("Listening...")

        n_packets = 0

        while True:

            #start_time = time.time()
            #data = bytearray(b'')
            try:
                print("waiting...")
                data = self.server_socket.recv(1024)
                print(list(data))
                print(type(data))
                n_packets += 1
                print("N PACKETS:{}".format(n_packets))
                packet = pickle.loads(data)
                handle(packet,handle_args)

            except Exception as e:
                print(e)
                self.server_socket.close()
                self.setup_server()
            '''

            while (len(data)<package_size):
                data.extend(self.server_socket.recv(VIDEO_STREAM_RECV_SIZE))

            print(data)

            packed_msg_size = data[:package_size]

            data = data[package_size:]

            msg_size = struct.unpack("L",packed_msg_size)[0]

            while (len(data) < msg_size):
                data.extend(self.server_socket.recv(VIDEO_STREAM_RECV_SIZE))


            if len(data):

                packet = pickle.loads(data)

                client_message = handle(packet,handle_args)
                
                n_packets += 1
                print("Recieved packets: {}".format(n_packets))

                #.send(client_message)
            '''
