import socket
import select
from datetime import datetime as dt
import argparse
from combine import load_model
from engine import translate
from seq2seq import Encoder, Decoder
from vocab import Lang

HEADER_LENGTH = 10

parser = argparse.ArgumentParser()
parser.add_argument("-host", "--host_ip", type=str, help="Host IP to start the server in", default='127.0.0.1')
parser.add_argument("-port", "--port_no", type=int, help="Port in which the server will start listening", default=1234)

args = parser.parse_args()

host = args.host_ip
port = args.port_no

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server_socket.bind((host, port))
server_socket.listen()
print(f"Server started at {dt.now().time()}")
sockets_list = [server_socket]

clients = {}

# Loading the model 
enc, dec, in_lang, out_lang = load_model('files/model_De')


def message_translate(msg):
	return translate(msg, enc, dec, in_lang, out_lang) 

def recieve_message(client_socket):
	try:
		message_header = client_socket.recv(HEADER_LENGTH)

		if not len(message_header):
			return False

		message_length = int(message_header.decode("utf-8").strip())
		return {"header": message_header, "data": client_socket.recv(message_length)}

	except:
		return False



while True:
	read_sockets, _, exception_sockets = select.select(sockets_list, [], sockets_list)

	for notified_socket in read_sockets:
		if notified_socket == server_socket:
			client_socket, client_address = server_socket.accept()

			user = recieve_message(client_socket)

			if user is False:
				continue

			language = recieve_message(client_socket)

			if language is False:
				language = 'en'
			sockets_list.append(client_socket)

			clients[client_socket] = user, language 

			print(f"[{dt.now().time()}] Accepted new connection from {client_address[0]}:{client_address[1]}")
			print(f"Username:{user['data'].decode('utf-8')} Language:{language['data'].decode('utf-8')}")

		else:
			message = recieve_message(notified_socket)

			if message is False:
				print(f"[{dt.now().time()}] Closed connection {clients[notified_socket][0]['data'].decode('utf-8')}")
				sockets_list.remove(notified_socket)
				del clients[notified_socket]
				continue
			
			user = clients[notified_socket][0]
			lang = clients[notified_socket][1]['data'].decode('utf-8')
			new_message = message['data'].decode('utf-8')

			print(f"[{dt.now().time()}] Recived message from {user['data'].decode('utf-8')}<{lang}>: {new_message}")
			
			for client_socket in clients:
				if client_socket != notified_socket:
					if lang != 'en':
						msg = ''
						try:
							msg = message_translate(new_message)
						except Exception:
							msg = new_message
						msg = msg.encode("utf-8")
						msg_header = f"{len(msg) :<{HEADER_LENGTH}}".encode("utf-8")
						client_socket.send(user['header'] + user['data'] + msg_header + msg)
					else:
						client_socket.send(user['header'] + user['data'] + message['header'] + message['data'])

	for notified_socket in exception_sockets:
		sockets_list.remove(notified_socket)
		del clients[notified_socket]