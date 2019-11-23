#!/usr/bin/env python2
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
import argparse
import os
import pickle
import sys
import numpy as np
np.set_printoptions(precision=2)
import openface
import ec_elgamal
import socket
import struct
import random
import pickle
from numpy import linalg as LA
from multiprocessing import Pool
from random import randint

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.", default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.", default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int, help="Default image dimension.", default=96)
parser.add_argument('--serverPort', type=int, help="Port of the server.", default=6546)
parser.add_argument('--serverIP', type=str, help="IP address of the server.", default="127.0.0.1")
parser.add_argument('--threshold', type=float, default=0.99)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--captureDevice', type=int, default=0, help='Capture device. 0 for latop webcam and 1 for usb webcam')
parser.add_argument('--width', type=int, default=320)
parser.add_argument('--height', type=int, default=240)
parser.add_argument('--load', action='store_true')
parser.add_argument('--CPUs', type=int, help="Number of parallel CPUs to be used.", default=4)
parser.add_argument('--Gportion', type=int, help="Portion of G to be stored locally (0: no G, 1:full G, 2:half G, 3:third G, 4:quarter G.", default=1)
parser.add_argument('--maxRam', type=int, help="Maximum amount of ram to be used by the system (in GB).", default=5)
args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)

# System parameters
server_port = args.serverPort
server_ip = args.serverIP
similarity_threshold = args.threshold
verbose = args.verbose
load = args.load
ec_elgamal_ct_size = 130
normalizing_adder = 128
normalizing_multiplier = 400
enc_similarity_threshold = ""
rand_nbrs_min_bitlen = 11
rand_nbrs_max_bitlen = 11
pub_key_file = "rec_pub.txt"
B_file = "rec_B.data"
C_file = "rec_C.data"
F_file = "F.data"
G_file = "G.data"
rand_numbers_file = "rand_num.data"
transition_time = 2	# (seconds)
nbr_of_CPUs = args.CPUs
max_ram = args.maxRam
G_portion = args.Gportion

# Temporary global variables
B = []
C = []
F = []
G = []
imgrep = []

def getRep(bgrImg):
    if bgrImg is None:
        return None
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    # bb = align.getLargestFaceBoundingBox(rgbImg)
    bb = align.getAllFaceBoundingBoxes(rgbImg)
    if bb is None:
        return None
    alignedFaces = []
    for box in bb:
        alignedFaces.append(align.align(args.imgDim, rgbImg, box, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))
    if alignedFaces is None:
        return None
    reps = []
    for alignedFace in alignedFaces:
        reps.append(net.forward(alignedFace))
    return (reps,bb)

def connectToServer():
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_address = (server_ip, server_port)
	if verbose:	print("connectToServer: Connecting to {}:{}...".format(server_ip, server_port))
	sock.connect(server_address)
	if verbose:	print("connectToServer: Connected")
	return sock

def getPubKey(sock):
	try:
		if verbose:	print("getPubKey: Getting the server's key...")
		message = 'GET pub_key'
		sock.sendall(message)
		pub_key = recv_msg(sock)
		f = open(pub_key_file, "w")
		f.write(pub_key)
		f.close()
		if verbose:	print("getPubKey: Key received")
	except:
		print("getPubKey: Error")

def getBCfiles(sock):
	try:
		if verbose:	print("getBCfiles: Getting matrices B and C...")
		message = "GET DBfiles"
		sock.sendall(message)
		B = recv_msg(sock)
		C = recv_msg(sock)
		f = open(B_file, 'w')
		f.write(B)
		f.close()
		f = open(C_file, 'w')
		f.write(C)
		f.close()
		if verbose:	print("getBCfiles: B and C received")
	except:
		print("getBCfiles: Error")

def encryptForF(list):
	return [ec_elgamal.encrypt_ec(str(i)) for i in list]

def encryptForG(list):
	return [[[ec_elgamal.mult(str(k), Cij) for k in range (256)] for Cij in Ci] for Ci in list]

def encryptForHalfG(list):
	return [[[ec_elgamal.mult(str(k), Cij) for k in range (64,192)] for Cij in Ci] for Ci in list]

def encryptForThirdG(list):
	return [[[ec_elgamal.mult(str(k), Cij) for k in range (85,171)] for Cij in Ci] for Ci in list]

def encryptForForthG(list):
	return [[[ec_elgamal.mult(str(k), Cij) for k in range (96,160)] for Cij in Ci] for Ci in list]

def generateLocalFiles():
	start_gen_files = time.time()
	ec_elgamal.prepare_for_enc(pub_key_file)

	if verbose:	print("generateLocalFiles: Generating vector F...")
	pool = Pool(processes=nbr_of_CPUs)
	F_values = [k**2 for k in range(256)]
	F = pool.map(encryptForF, (F_values[int(i*256/nbr_of_CPUs):int((i+1)*256/nbr_of_CPUs)] for i in range(nbr_of_CPUs)))
	F = [ent for sublist in F for ent in sublist]
	np.save(F_file, F)
	if verbose:	print("generateLocalFiles: F generated")
	del F

	global G_portion
	G = []
	C = np.load(C_file)
	suspects_count = len(C)
	if verbose:	print("generateLocalFiles: Generating matrix G for {} suspects...".format(suspects_count))
	expected_G_size = os.path.getsize(C_file)*256
	if G_portion==1:
		G = pool.map(encryptForG, (C[int(i*suspects_count/nbr_of_CPUs):int((i+1)*suspects_count/nbr_of_CPUs)] for i in range(nbr_of_CPUs)))
	elif G_portion==2:
		G = pool.map(encryptForHalfG, (C[int(i*suspects_count/nbr_of_CPUs):int((i+1)*suspects_count/nbr_of_CPUs)] for i in range(nbr_of_CPUs)))
	elif G_portion==3:
		G = pool.map(encryptForThirdG, (C[int(i*suspects_count/nbr_of_CPUs):int((i+1)*suspects_count/nbr_of_CPUs)] for i in range(nbr_of_CPUs)))
	elif G_portion==4:
		G = pool.map(encryptForForthG, (C[int(i*suspects_count/nbr_of_CPUs):int((i+1)*suspects_count/nbr_of_CPUs)] for i in range(nbr_of_CPUs)))

	if G != []:
		G = [ent for sublist in G for ent in sublist]
		np.save(G_file, G)
		del G
	else:
		G_portion = 0
	pool.close()
	end_gen_files = time.time()
	if verbose:	print("generateLocalFiles: G generated")
	print("generateLocalFiles: Local files F & G have been computed in {}s".format((end_gen_files-start_gen_files)*1000))

	results_file = open("final_results.txt","a+")
	storage = ec_elgamal_ct_size*(128*suspects_count*256/G_portion+128*suspects_count+256+suspects_count)+256*suspects_count if G_portion>0 else ec_elgamal_ct_size*(128*suspects_count+128*suspects_count+256+suspects_count)+256*suspects_count
	storage = storage*1.00/1024/1024
	results_file.write("Offile:M= {} CPUs_camera= {} F_G_gen= {} G_portion= {} storage((GorC)+B+F+rand)= {}\n".format(suspects_count,nbr_of_CPUs,end_gen_files-start_gen_files,G_portion,storage))
	results_file.close()

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

def normalizeRep(rep):
	normalizedRep = [int(x*normalizing_multiplier+normalizing_adder) for x in rep]
	return normalizedRep

def fill_rand_numbers_file():
	B = np.load(B_file)
	suspects_count = len(B)
	ec_elgamal.prepare_for_enc(pub_key_file)
	f = open(rand_numbers_file, "wb")
	for i in range(suspects_count):
		rand1 = random.getrandbits(randint(rand_nbrs_min_bitlen, rand_nbrs_max_bitlen))
		rand2 = random.getrandbits(randint(rand_nbrs_min_bitlen, rand_nbrs_max_bitlen))
		if (rand1 < rand2):
			rand1, rand2 = rand2, rand1		# always rand1 > rand2
		f.write(str(rand1)+'\n')
		f.write(ec_elgamal.encrypt_ec(str(rand2)))
	f.close()

def sendScores(connection, scores):
	try:
		message = "new_scores "
		connection.sendall(message)
		send_msg(connection, scores)
	except:
		print ("sendScores: Error")

def compute_sub_scores(list):
	D = []
	if G_portion==0:
		for i in list:
			Bi = B[i]
			Ci = C[i]
			enc_d = ec_elgamal.add3(Bi[0], F[imgrep[0]], ec_elgamal.mult(str(imgrep[0]), Ci[0]))
			for j in range(1, 128):
				enc_d = ec_elgamal.add4(enc_d, Bi[j], F[imgrep[j]], ec_elgamal.mult(str(imgrep[j]), Ci[j]))
			D.append(enc_d)
	elif G_portion==1:
		for i in list:
			Bi = B[i]
			Gi = G[i]
			enc_d = ec_elgamal.add3(Bi[0], F[imgrep[0]], Gi[0][imgrep[0]])
			for j in range(1, 128):
				enc_d = ec_elgamal.add4(enc_d, Bi[j], F[imgrep[j]], Gi[j][imgrep[j]])
			D.append(enc_d)
	else:
		first_index = 64 if G_portion==2 else 85 if G_portion==3 else 96 if G_portion==4 else 0
		last_index = 192 if G_portion==2 else 171 if G_portion==3 else 160 if G_portion==4 else 0
		for i in list:
			Bi = B[i]
			Ci = C[i]
			Gi = G[i]
			if first_index <= imgrep[0] < last_index:
				enc_d = ec_elgamal.add3(Bi[0], F[imgrep[0]], Gi[0][imgrep[0]-first_index])
			else:
				enc_d = ec_elgamal.add3(Bi[0], F[imgrep[0]], ec_elgamal.mult(str(imgrep[0]), Ci[0]))
			for j in range(1, 128):
				if first_index <= imgrep[j] < last_index:
					enc_d = ec_elgamal.add4(enc_d, Bi[j], F[imgrep[j]], Gi[j][imgrep[j]-first_index])
				else:
					enc_d = ec_elgamal.add4(enc_d, Bi[j], F[imgrep[j]], ec_elgamal.mult(str(imgrep[j]), Ci[j]))
			D.append(enc_d)
	return D

if __name__ == '__main__':
	global G_portion
	# Offline phase___________________________________________
	sock = connectToServer()

	if not load:
		getPubKey(sock)
		if verbose:	print("main:pub_key received and saved in {}".format(pub_key_file))
		getBCfiles(sock)
		if verbose:	print("main:server files received and saved in {}, {}".format(B_file, C_file))
		generateLocalFiles()
		if verbose:	print("main:local files have been generated")
		fill_rand_numbers_file()
		if verbose:	print("main:random files file {} has been filled".format(rand_numbers_file))

	# Online phase____________________________________________
	ec_elgamal.prepare_for_enc(pub_key_file)

	enc_similarity_threshold = ec_elgamal.encrypt_ec(str(-(similarity_threshold*normalizing_multiplier)**2))

	B = np.load(B_file)
	suspects_count = len(B)
	F = np.load(str(F_file+".npy"))
	if G_portion != 1:
		C = np.load(C_file)
	if G_portion != 0:
		G = np.load(str(G_file+".npy"))
		#Get G_portion from the stored G
		G_portion = 1 if len(G[0][0])==256 else 2 if len(G[0][0])==128 else 3 if len(G[0][0])==86 else 4 if len(G[0][0])==64 else 0
	print("main: G_portion={}".format(G_portion))

	# initialize the camera and grab a reference to the raw camera capture
	camera = PiCamera()
	camera.resolution = (args.width, args.height)
	camera.framerate = 32
	rawCapture = PiRGBArray(camera, size=(args.width, args.height))
	# allow the camera to warmup
	time.sleep(0.1)
	
	persons_reps = []
	start_time = time.time()

	# capture frames from the camera
	for lframe in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
		start_comp_time = time.time()
		frame = lframe.array
		#cv2.imshow(" ", frame)
		rawCapture.truncate(0)

		repsAndBBs = getRep(frame)
		reps = repsAndBBs[0]
		bbs = repsAndBBs[1]
		normalized_reps = [normalizeRep(face) for face in reps]
		reps_array = np.array(normalized_reps)

		if (len(reps) == 0):	# no face in frame
			continue
		person_id = -1
		for idx, rep in enumerate(persons_reps):
			dist = LA.norm(reps_array-rep)
			if dist < similarity_threshold*normalizing_multiplier :
				person_id = idx
				start_time = time.time()
				break
		if person_id == -1:		# if a new person has been detected
			now = time.time()
			if (now - start_time < transition_time):
				continue

			start_enc_time = time.time()
			persons_reps.append(reps_array)
			person_id += 1
			imgrep = reps_array[0]
	
			if verbose:	print("main: Computing distances...")
			dist_comp_start = time.time()
			pool = Pool(processes=nbr_of_CPUs)
			suspects_indices = range(suspects_count)
			D = pool.map(compute_sub_scores, (suspects_indices[int(i*suspects_count/nbr_of_CPUs):int((i+1)*suspects_count/nbr_of_CPUs)] for i in range(nbr_of_CPUs)))
			D = [ent for sublist in D for ent in sublist]
			pool.close()
			dist_comp_end = time.time()
			if verbose:	print("main: Distances computation = {} ms".format((dist_comp_end-dist_comp_start)*1000))

			if verbose:	print("main: Obfuscating distances...")
			dist_obf_start = time.time()
			f = open(rand_numbers_file, "rb")
			D = [ec_elgamal.add2(ec_elgamal.mult(f.readline(), ec_elgamal.add2(D[i], enc_similarity_threshold)), f.read(ec_elgamal_ct_size)) for i in range(suspects_count)]
			f.close()
			dist_obf_end = time.time()
			if verbose:	print("main: Distances obfuscation = {} ms".format((dist_obf_end-dist_obf_start)*1000))

			dist_comp_obf_end = time.time()
			if verbose:	print("main:time(face recognition + scores computation + scores obfuscation) for {} suspects: {} ms".format(suspects_count, (dist_comp_obf_end-start_comp_time)*1000))
			if verbose:	print("main:enc time for {} suspects: {} ms".format(suspects_count, (dist_comp_obf_end-start_enc_time)*1000))

			results_file = open("final_results.txt","a+")
			results_file.write("Online:M= {} CPUs_camera= {} dist_comp= {} dist_obf= {} total_time= {} enc_time= {}\n".format(suspects_count,nbr_of_CPUs,(dist_comp_end-dist_comp_start)*1000,(dist_obf_end-dist_obf_start)*1000,(dist_comp_obf_end-start_comp_time)*1000,(dist_comp_obf_end-start_enc_time)*1000))
			results_file.close()
			
			if verbose:	print("main: Sending encrypted scores to the server...")
			enc_D = pickle.dumps(D)
			sendScores(sock, enc_D)
			print("main: Encrypted scores sent")
			data = sock.recv(11)
			if (data == "GET image  "):
				# TODO blure irrelevant faces in the frame using bbs
				print("main: Suspect detected! Sending suspect's image to the server...")
				image = pickle.dumps(frame)
				send_msg(sock, image)
				end_comp_time = time.time()
				if verbose:	print("main:suspect detected: total rtt: {} ms".format((end_comp_time-start_comp_time)*1000))
				
				results_file = open("final_results.txt","a+")
				results_file.write("Online:RTT= {}\n".format(end_comp_time-start_comp_time))
				results_file.close()

		# for idx,person in enumerate(reps):
		# 	cv2.rectangle(frame, (bbs[idx].left(), bbs[idx].top()), (bbs[idx].right(), bbs[idx].bottom()), (0, 255, 0), 2)
		# 	# cv2.putText(frame, "{} @{:.2f}".format(person, confidences[idx]), (bbs[idx].left(), bbs[idx].bottom()+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
		# 	cv2.putText(frame, "person {}/{}".format(person_id, len(persons_reps)), (bbs[idx].left(), bbs[idx].bottom()+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
		# cv2.imshow('', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
