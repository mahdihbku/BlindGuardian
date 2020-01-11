#!/usr/bin/env python2
import time
import argparse
import cv2
import os
import pickle
import sys
import numpy as np
np.set_printoptions(precision=2)
import openface
import ec_elgamal
import socket
import struct
import datetime
from multiprocessing import Pool

fileDir				= os.path.dirname(os.path.realpath(__file__))
modelDir			= os.path.join(fileDir, '..', 'models')
suspectsDir			= os.path.join(fileDir, '..', 'suspectsDir')
dlibModelDir		= os.path.join(modelDir, 'dlib')
openfaceModelDir	= os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()
parser.add_argument('--suspectsDir',		type=str,	help="Path to directory containing the webcam faces.",	default=suspectsDir															)
parser.add_argument('--dlibFacePredictor',	type=str,	help="Path to dlib's face predictor.",					default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")	)
parser.add_argument('--networkModel',		type=str,	help="Path to Torch network model.",					default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')					)
parser.add_argument('--imgDim',				type=int,	help="Default image dimension.",						default=96																	)
parser.add_argument('--serverPort',			type=int,	help="Port of the server.",								default=6546																)
parser.add_argument('--serverIP',			type=str,	help="IP of the server.",								default='127.0.0.1'															)
parser.add_argument('--CPUs',				type=int,	help="Number of parallel CPUs to be used.",				default=4																	)
parser.add_argument('--verbose',						help="Show more information.", 							action='store_true'															)
parser.add_argument('--generateKeys',					help="Generate new server keys.", 						action='store_true'															)
args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)

# System parameters
server_ip = args.serverIP
server_port = args.serverPort
nbr_of_CPUs = args.CPUs
verbose = args.verbose
normalizing_adder = 128
normalizing_multiplier = 400
pub_key_file = "ec_pub.txt"
priv_key_file = "ec_priv.txt"
B_file = 'B.data'
C_file = 'C.data'
suspects_names = []

def getRep(imgPath):
	bgrImg = cv2.imread(imgPath)
	if bgrImg is None:
		return []
	rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
	bb = align.getLargestFaceBoundingBox(rgbImg)
	if bb is None:
		return []
	alignedFace = align.align(args.imgDim, rgbImg, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
	if alignedFace is None:
		return []
	rep = net.forward(alignedFace)
	return rep

def sendPubKey(connection):
	f = open(pub_key_file, 'r')
	pub_key = f.read()
	send_msg(connection, pub_key)
	if verbose:	print("sendPubKey: Public key sent")
	f.close()

def sendBCfiles(connection):
	f = open(str(B_file+".npy"), "rb")
	B = f.read()
	send_msg(connection, B)
	if verbose:	print("sendBCfiles: B sent")
	f.close()
	f = open(str(C_file+".npy"), "rb")
	C = f.read()
	send_msg(connection, C)
	if verbose:	print("sendBCfiles: C sent")
	f.close()

def decryptList(enc_list):
	return [ec_elgamal.decrypt_ec(d) for d in enc_list]

def isScorePositiveList(enc_list):
	return [ec_elgamal.score_is_positive(d) for d in enc_list]

def getScores(connection):
	try:
		ec_elgamal.prepare(pub_key_file, priv_key_file)
		data = recv_msg(connection)
		enc_D = pickle.loads(data)

		start_dec = time.time()
		D = [ec_elgamal.score_is_positive(encrypted_score) for encrypted_score in enc_D] #TODO uncomment
		# pool = Pool(processes=nbr_of_CPUs)
		# D = pool.map(isScorePositiveList, (enc_D[int(i*len(enc_D)/nbr_of_CPUs):int((i+1)*len(enc_D)/nbr_of_CPUs)] for i in range(nbr_of_CPUs)))
		# pool.close()
		# D = [ent for sublist in D for ent in sublist]	#to flatten the resultant Ds into one D
		
		# D = [ec_elgamal.decrypt_ec(encrypted_score) for encrypted_score in enc_D] #TODO comment
		end_dec = time.time()
		if verbose:	print("getScores: dec_time for {} suspects: {} ms.".format(len(D), (end_dec-start_dec)*1000))
		# if verbose:	print(D) #TODO comment

		results_file = open("final_results.txt", "a+")
		results_file.write("Online:dec_time= {}\n".format((end_dec-start_dec)*1000))
		results_file.close()

		if (0 in D):	# SUSPECT DETECTED!!!
			suspect_id = D.index(0)
			print("getScores: SUSPECT DETECTED! id={} name={}".format(suspect_id, suspects_names[suspect_id]))
			message = "GET image  "
			connection.sendall(message)
			data = recv_msg(connection)
			now = datetime.datetime.now()
			image_name = "suspect"+str(now.strftime("%Y-%m-%d-%H-%M")+".png")
			frame = pickle.loads(data)
			cv2.imwrite(image_name, frame)
			print("getScores: Suspect's image saved in {}".format(image_name))
		else:
			message = "No match   "
			connection.sendall(message)
	except:
		print 'getScores: Error'

def waitForClients():
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_address = (server_ip, server_port)
	sock.bind(server_address)
	sock.listen(10)
	while True:
		# Wait for a connection
		print("waitForClients: Waiting for a connection")
		connection, client_address = sock.accept()
		try:
			while True:
				data = recvall(connection, 11)
				print("waitForClients: Received: {}".format(data))
				if data:
					if (data == "GET pub_key"):
						sendPubKey(connection)
					if (data == "GET DBfiles"):	# B and C
						sendBCfiles(connection)
					if (data == "new_scores "):
						getScores(connection)
				else:
					print("waitForClients: No more data from client {}".format(client_address))
					break
		finally:
			connection.close()
			print("waitForClients: Connection closed with client {}".format(client_address))

def normalizeRep(rep):
	normalized_rep = [int(r*normalizing_multiplier+normalizing_adder) for r in rep]
	for idx in range(len(rep)):
		if normalized_rep[idx] > 255:	normalized_rep[idx] = 255
		elif normalized_rep[idx] < 0:	normalized_rep[idx] = 0
	return normalized_rep

def encryptForB(list):
	return [ec_elgamal.encrypt_ec(str(sum([f**2 for f in suspect]))) for suspect in list]

def encryptForC(list):
	return [[ec_elgamal.encrypt_ec(str(-2*f)) for f in suspect] for suspect in list]

def generateDBfiles():
	start_norm = time.time()
	ec_elgamal.prepare(pub_key_file, priv_key_file)
	suspects_reps = []
	suspects_names[:] = []
	if verbose:	print("generateDBfiles: Detecting and normalizing faces...")
	for root, dirs, files in os.walk(args.suspectsDir):
		for img in files:
			imgrep = getRep(os.path.join(root, img))
			if len(imgrep) == 0: continue
			suspects_reps.append(normalizeRep(imgrep))
			suspects_names.append(os.path.join(root, img))
	end_norm = time.time()
	if verbose:	print("generateDBfiles: Suspects in dir {} have been normalized in {}".format(args.suspectsDir, (end_norm-start_norm)*1000))

	start_enc = time.time()
	if verbose:	print("generateDBfiles: Generating matrix B...")
	pool = Pool(processes=nbr_of_CPUs)
	B = pool.map(encryptForB, (suspects_reps[int(i*len(suspects_reps)/nbr_of_CPUs):int((i+1)*len(suspects_reps)/nbr_of_CPUs)] for i in range(nbr_of_CPUs)))
	B = [ent for sublist in B for ent in sublist]
	if verbose:	print("generateDBfiles: B generated for {} faces".format(len(B)))
	if verbose:	print("generateDBfiles: Generating matrix C...")
	C = pool.map(encryptForC, (suspects_reps[int(i*len(suspects_reps)/nbr_of_CPUs):int((i+1)*len(suspects_reps)/nbr_of_CPUs)] for i in range(nbr_of_CPUs)))
	pool.close()
	C = [ent for sublist in C for ent in sublist]
	if verbose:	print("generateDBfiles: C generated")
	end_enc = time.time()
	print("generateDBfiles: DB files generated in: {} ms.".format((end_enc-start_enc)*1000))

	results_file = open("final_results.txt", "a+")
	results_file.write("Offline:M= {} CPUs_srvr= {} ident+norm= {} BCgen= {} storage(B+C+keys)= {} off_comm= {} onl_comm= {}\n".format(len(suspects_reps), nbr_of_CPUs, end_norm-start_norm, end_enc-start_enc, 2*len(suspects_reps)*128*512*1.00/1024/1024, 2*len(suspects_reps)*128*512*1.00/1024/1024, len(suspects_reps)*512*1.00/1024))
	results_file.close()

	np.save(B_file, B)
	np.save(C_file, C)

def send_msg(sock, msg):
	# Prefix each message with a 4-byte length (network byte order)
	msg = struct.pack('>I', len(msg)) + msg
	sock.sendall(msg)

def recv_msg(sock):
	# Read message length and unpack it into an integer
	raw_msglen = recvall(sock, 4)
	if not raw_msglen:
		return None
	msglen = struct.unpack('>I', raw_msglen)[0]
	# Read the message data
	return recvall(sock, msglen)

def recvall(sock, n):
	# Helper function to recv n bytes or return None if EOF is hit
	data = b''
	while len(data) < n:
		packet = sock.recv(n - len(data))
		if not packet:
			return None
		data += packet
	return data

if __name__ == '__main__':
	#if args.generateKeys :	# to use with precaution. That will require regenerating the decryption file
		#ec_elgamal.generate_keys(pub_key_file, priv_key_file)
		#TODO uncomment
		#ec_elgamal.prepare(pub_key_file, priv_key_file)
		#ec_elgamal.generate_decrypt_file()
	print("main: Loading the decryption file to memory, this may take few minutes...")
	ec_elgamal.load_encryption_file()
	generateDBfiles()
	waitForClients()
