import socket
import select
import errno
import sys
import argparse
from datetime import datetime as dt
import winsound

HEADER_LENGTH = 10

parser = argparse.ArgumentParser()
parser.add_argument("-host", "--host_ip", type=str, help="Host IP to connect to the ", default='127.0.0.1')
parser.add_argument("-port", "--port_no", type=int, help="Port of the server to connect to", default=1234)

args = parser.parse_args()

host = args.host_ip
port = args.port_no

m_recvd_file = 'files/beep.wav'

my_username = input("Username: ")
my_language = input("Language(En/DE):").lower()

while my_language != 'en' and my_language != 'de':
	print('\nSorry but we currently support two languages only!')
	print('[En: English and De: German]\n')
	my_language = input("Language(En/DE):").lower()


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((host, port))
client_socket.setblocking(False)

username = my_username.encode("utf-8")
username_header = f"{len(username) :<{HEADER_LENGTH}}".encode("utf-8")
client_socket.send(username_header + username)

language = my_language.encode("utf-8")
language_header = f"{len(language) :<{HEADER_LENGTH}}".encode("utf-8")
client_socket.send(language_header + language)

uname = '[YOU]'
while True:
	message = input(f"{uname :^{HEADER_LENGTH}} > ")
	
	if message:
		message = message.encode("utf-8")
		message_header = f"{len(message) :<{HEADER_LENGTH}}".encode('utf-8')
		client_socket.send(message_header + message)

	try:
		while True:
			# Recieve THings
			username_header = client_socket.recv(HEADER_LENGTH)
			if not len(username_header):
				print("Connection closed by the server!")
				sys.exit()

			username_length = int(username_header.decode("utf-8").strip())
			username = client_socket.recv(username_length).decode('utf-8')

			message_header = client_socket.recv(HEADER_LENGTH)
			message_length = int(message_header.decode("utf-8").strip())
			message = client_socket.recv(message_length).decode("utf-8")

			print(f"{username :^{HEADER_LENGTH}} > {message}")
			winsound.PlaySound(m_recvd_file, winsound.SND_FILENAME)

	except IOError as e:
		if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
			print("Reading Error: ", str(e))
			sys.exit()
		continue

	except Exception as e:
		print('General error: ', str(e))
		sys.exit()

		