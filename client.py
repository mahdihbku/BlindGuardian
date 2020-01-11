#!/usr/bin/env python2
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
from random import randint
from multiprocessing import Pool
from multiprocessing import Process, JoinableQueue, Lock, Manager

fileDir 			= os.path.dirname(os.path.realpath(__file__))
modelDir 			= os.path.join(fileDir, '..', 'models')
dlibModelDir 		= os.path.join(modelDir, 'dlib')
openfaceModelDir	= os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()
#Capturing options in order:
parser.add_argument('--PiCamera',			action='store_true',		help='Execute client.py from Raspberry Pi.')
parser.add_argument('--oneImage',			type=str,	default="",		help="Send the scores for one image.")
parser.add_argument('--video',				type=str,	default="",		help="Detect faces from a video sequence.")
parser.add_argument('--captureDevice',		type=int,	default=0,		help='Capture device. 0 for latop webcam and 1 for usb webcam.')
#Other parameters:
parser.add_argument('--dlibFacePredictor',	type=str,	default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"),	help="Path to dlib's face predictor.")
parser.add_argument('--networkModel',		type=str,	default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'),	help="Path to Torch network model.")
parser.add_argument('--imgDim',				type=int,	default=96,		help="Default image dimension.")
parser.add_argument('--serverPort',			type=int,	default=6546,	help="Port of the server.")
parser.add_argument('--serverIP',			type=str,	default="127.0.0.1", help="IP address of the server.")
parser.add_argument('--threshold',			type=float,	default=0.99,	help="Similarity threshold.")
parser.add_argument('--verbose',			action='store_true',		help="Show more details(execution steps, times...).")
parser.add_argument('--width',				type=int,	default=640,	help='Width of frame.')
parser.add_argument('--height',				type=int,	default=480,	help='Height of frame.')
parser.add_argument('--load',				action='store_true',		help='Load stored server information and database.')
parser.add_argument('--CPUs',				type=int,	default=4,		help="Number of parallel CPUs to be used.")
parser.add_argument('--Gportion',			type=int,	default=1,		help="Portion of G to be stored locally (0: no G, 1:full G, 2:half G, 3:third G, 4:quarter G.")
parser.add_argument('--maxRam',				type=int,	default=5,		help="Maximum amount of ram to be used by the system (in GB).")
args = parser.parse_args()

# System parameters
align = openface.AlignDlib(args.dlibFacePredictor)
#net = openface.TorchNeuralNet(args.networkModel, args.imgDim)
server_port				= args.serverPort
server_ip				= args.serverIP
similarity_threshold	= args.threshold
verbose					= args.verbose
load					= args.load
ec_elgamal_ct_size		= 130
normalizing_adder		= 128		#normalizing parameter
normalizing_multiplier	= 400		#normalizing parameter
enc_similarity_threshold= ""
rand_nbrs_min_bitlen	= 11
rand_nbrs_max_bitlen	= 11
pub_key_file			= "rec_pub.txt"
B_file					= "rec_B.data"
C_file					= "rec_C.data"
# F_file					= "F.data"
G_file					= "G.data"
rand_numbers_file		= "rand_num.data"
nbr_of_CPUs				= args.CPUs
max_ram					= args.maxRam
G_portion				= args.Gportion
Pi_camera				= args.PiCamera
#transition_time		= 2	# (seconds)

# Temporary global variables
B			= []	#db file to receive from server
C			= []	#db file to receive from server
G			= []	#generated locally from C
r1_list		= []	#list of randomly generated r1
r2_list		= []	#list of randomly generated r2
first_index	= 0		#starting index stored in G
last_index	= 256	#last index stored in G
queue		= JoinableQueue(maxsize=nbr_of_CPUs)	
lock		= Lock()	# mutex to protect persons_reps
manager		= Manager()
persons_reps= manager.list()	# detected faces by the camera

def getRep(bgrImg, net):	#get all face bouding boxes
	if bgrImg is None:
		return None
	rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
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
	sock.connect(server_address)
	return sock

def getPubKey(sock):
	try:
		message = 'GET pub_key'
		sock.sendall(message)
		pub_key = recv_msg(sock)
		f = open(pub_key_file, "w")
		f.write(pub_key)
		f.close()
	except:
		print 'getPubKey: Error getting pub key from server'

def getBCfiles(sock):
	try:
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
	except:
		print 'getBCfiles: Error getting matrices B and C from server'

def encryptForG(list):
	return [[[ec_elgamal.mult(str(k), Cij) for k in range (first_index, last_index)] for Cij in Ci] for Ci in list]

def generateLocalFiles():
	global G_portion
	global first_index
	global last_index
	start_gen_files = time.time()
	ec_elgamal.prepare_for_enc(pub_key_file)
	# ec_elgamal.prepare(pub_key_file, "priv.txt")	#for debug purpose only

	C = np.load(C_file)
	suspects_count = len(C)
	expected_G_size = os.path.getsize(C_file)*256
	#TODO compare expected_G_size with available RAM
	first_index = 0  if G_portion==1 else 64  if G_portion==2 else 85  if G_portion==3 else 96  if G_portion==4 else 0
	last_index = 256 if G_portion==1 else 192 if G_portion==2 else 171 if G_portion==3 else 160 if G_portion==4 else 0
	pool = Pool(processes=nbr_of_CPUs)
	G = pool.map(encryptForG, (C[int(i*suspects_count/nbr_of_CPUs):int((i+1)*suspects_count/nbr_of_CPUs)] for i in range(nbr_of_CPUs)))
	pool.close()
	if G != []:
		G = [ent for sublist in G for ent in sublist]
		np.save(G_file, G)
		del G
	else:
		G_portion = 0
	end_gen_files = time.time()

	if verbose:	print("generateLocalFiles: Local files have been generated in {}".format((end_gen_files-start_gen_files)*1000))

	results_file = open("final_results.txt","a+")
	storage = ec_elgamal_ct_size*(128*suspects_count*256/G_portion+128*suspects_count+256+suspects_count)+256*suspects_count if G_portion>0 else ec_elgamal_ct_size*(128*suspects_count+128*suspects_count+256+suspects_count)+256*suspects_count
	results_file.write("Offile:M= {} CPUs_camera= {} F_G_gen= {} G_portion= {} storage((GorC)+B+F+rand)= {}\n".format(suspects_count, nbr_of_CPUs, end_gen_files-start_gen_files, G_portion, storage*1.00/1024/1024))
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
	for idx in range(len(rep)):
		if normalizedRep[idx] > 255:	normalizedRep[idx] = 255
		elif normalizedRep[idx] < 0:	normalizedRep[idx] = 0
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
		print 'sendScores:error'

def compute_sub_scores(list, imgrep):
	D = []
	enc_y2 = ec_elgamal.encrypt_ec(str(sum([y**2 for y in imgrep])))
	if G_portion == 0:
		for i in list:
			enc_d = ec_elgamal.add2(enc_y2, B[i])	# B_i = sum_j(x_ij^2)
			Ci = C[i]
			for j in range(128):
				enc_d = ec_elgamal.add2(enc_d, ec_elgamal.mult(str(imgrep[j]), Ci[j]))
			D.append(enc_d)
	elif G_portion == 1:
		for i in list:
			enc_d = ec_elgamal.add2(enc_y2, B[i])	# B_i = sum_j(x_ij^2)
			Gi = G[i]
			for j in range(128):
				enc_d = ec_elgamal.add2(enc_d, Gi[j][imgrep[j]])
			D.append(enc_d)
	else:
		for i in list:
			enc_d = ec_elgamal.add2(enc_y2, B[i])	# B_i = sum_j(x_ij^2)
			Ci = C[i]
			Gi = G[i]
			for j in range(0, 128):
				if first_index <= imgrep[j] < last_index:
					enc_d = ec_elgamal.add2(enc_d, Gi[j][imgrep[j]-first_index])
				else:
					enc_d = ec_elgamal.add2(enc_d, ec_elgamal.mult(str(imgrep[j]), Ci[j]))
			D.append(enc_d)
	return D

def camThread():
	global persons_reps
	if verbose:	print("camThread: Thread started...")
	net = openface.TorchNeuralNet(os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'), args.imgDim)
	while True:
		frame = queue.get()
		start_comp_time = time.time()
		face_reco_start_time = time.time()
		if frame is None:
			queue.task_done()
			continue
		repsAndBBs = getRep(frame, net)
		reps = repsAndBBs[0]
		if (len(reps) == 0):	# no face in frame
			if verbose:	print("camThread: No face in the frame")
			queue.task_done()
			continue
		bbs = repsAndBBs[1]
		reps_array = np.array([normalizeRep(face) for face in reps])
		for face in reps_array:
			person_id = -1
			lock.acquire()
			#TODO right now. send all detected reps. not only the biggest!!!
			for idx, rep in enumerate(persons_reps):
				dist = LA.norm(face-rep)
				if dist < similarity_threshold*normalizing_multiplier :
					person_id = idx
					break
			if person_id != -1:
				if verbose:	print("camThread: No new face in the frame")
				lock.release()
				continue
			# if a new person has been detected
			persons_reps.append(face)
			lock.release()
			face_reco_end_time = time.time()
			print("camThread: A new face has been detected in the frame")
			if verbose:	print("camThread: Face recognition = {} ms".format((face_reco_end_time-face_reco_start_time)*1000))
			start_enc_time = time.time()
			dist_comp_start = time.time()
			person_id += 1
			#pool = Pool(processes=nbr_of_CPUs)
			#suspects_indices = range(suspects_count)
			#D = pool.map(compute_sub_scores, (suspects_indices[int(i*suspects_count/nbr_of_CPUs):int((i+1)*suspects_count/nbr_of_CPUs)] for i in range(nbr_of_CPUs)))
			#D = [ent for sublist in D for ent in sublist]
			#pool.close()
			D = compute_sub_scores(range(suspects_count), face)
			dist_comp_end = time.time()
			if verbose:	print("camThread: Computing the distance = {} ms".format((dist_comp_end-dist_comp_start)*1000))
			dist_obf_start = time.time()
			D = [ec_elgamal.add2(ec_elgamal.mult(r1_list[i], ec_elgamal.add2(D[i], enc_similarity_threshold)), r2_list[i]) for i in range(suspects_count)] #TODO uncomment
			# D = [ec_elgamal.add2(D[i], enc_similarity_threshold) for i in range(suspects_count)]	#Without obfuscation  #TODO comment
			dist_obf_end = time.time()
			if verbose:	print("camThread: Obfuscating the distance = {} ms".format((dist_obf_end-dist_obf_start)*1000))

			dist_comp_obf_end = time.time()
			if verbose:	print("camThread: Time(face recognition + scores computation + scores obfuscation) for {} suspects: {} ms".format(suspects_count, (dist_comp_obf_end-start_comp_time)*1000))
			if verbose:	print("camThread: Enc time for {} suspects: {} ms".format(suspects_count, (dist_comp_obf_end-start_enc_time)*1000))

			results_file = open("final_results.txt","a+")
			results_file.write("Online:M= {} CPUs_camera= {} dist_comp= {} dist_obf= {} total_time= {} enc_time= {}\n".format(suspects_count,nbr_of_CPUs,(dist_comp_end-dist_comp_start)*1000,(dist_obf_end-dist_obf_start)*1000,(dist_comp_obf_end-start_comp_time)*1000,(dist_comp_obf_end-start_enc_time)*1000))
			results_file.close()

			enc_D = pickle.dumps(D)
			sendScores(sock, enc_D)
			print("camThread: The encrypted scores have been sent to the server")
			data = sock.recv(11)
			if (data == "GET image  "):
				# sending the suspects image
				for idx in range(1,len(bbs)):	#ignore 0
					cv2.rectangle(frame, (bbs[idx].left(), bbs[idx].top()), (bbs[idx].right(), bbs[idx].bottom()), (255, 255, 255), -1)
				image = pickle.dumps(frame)
				send_msg(sock, image)
				print("camThread: Suspect detected! Irrelevant faces blured and the image has been sent")
				end_comp_time = time.time()
				if verbose:	print("camThread: Suspect detected: total rtt: {} ms".format((end_comp_time-start_comp_time)*1000))
				
				results_file = open("final_results.txt","a+")
				results_file.write("Online:RTT= {}\n".format(end_comp_time-start_comp_time))
				results_file.close()
		queue.task_done()
		if args.oneImage != "":
			break

if __name__ == '__main__':
	if verbose:	print("main: Connecting to server {}:{}...".format(server_ip, server_port))
	sock = connectToServer()
	if verbose:	print("main: Connected")

	# Offline phase___________________________________________
	if not load:
		if verbose:	print("main: Getting remote files...")
		getPubKey(sock)
		if verbose:	print("main: Pub_key received and saved in {}".format(pub_key_file))
		getBCfiles(sock)
		if verbose:	print("main: Server files received and saved in {}, {}".format(B_file, C_file))
		if verbose:	print("main: Generating local files (G_portion={})...".format(G_portion))
		generateLocalFiles()
		if verbose:	print("main: Local files have been generated and saved in {}".format(G_file))
		if verbose:	print("main: Generating random numbers file...")
		fill_rand_numbers_file()
		if verbose:	print("main: Random files file {} has been filled".format(rand_numbers_file))

	# Online phase____________________________________________
	if verbose:	print("main: Loading local variables to memory:")
	ec_elgamal.prepare_for_enc(pub_key_file)
	#ec_elgamal.prepare(pub_key_file, "ec_priv.txt") #for debugging only TODO to remove
	enc_similarity_threshold = ec_elgamal.encrypt_ec(str(-(similarity_threshold*normalizing_multiplier)**2))
	if verbose:	print("main:    Loading B...")
	B = np.load(B_file)
	suspects_count = len(B)
	if verbose:	print("main:    Loading rand numbers file...")
	file = open(rand_numbers_file, "rb")
	r1 = file.readline()
	while r1:
		r1_list.append(r1)
		r2 = file.read(ec_elgamal_ct_size)
		r2_list.append(r2)
		r1 = file.readline()
		if not r1:
			break
	file.close()
	
	if verbose:	print("main:    Loading G...")
	if G_portion != 0:
		G = np.load(str(G_file+".npy"))
		#Get G_portion from the stored G
		G_portion = 1 if len(G[0][0])==256 else 2 if len(G[0][0])==128 else 3 if len(G[0][0])==86 else 4 if len(G[0][0])==64 else 0
	if G_portion != 1:
		C = np.load(C_file)
	print("main:    G_portion={}".format(G_portion))
	first_index = 0  if G_portion==1 else 64  if G_portion==2 else 85  if G_portion==3 else 96  if G_portion==4 else 0
	last_index = 256 if G_portion==1 else 192 if G_portion==2 else 171 if G_portion==3 else 160 if G_portion==4 else 0
	if verbose:	print("main: Local variables have been loaded successfully")

	if Pi_camera:
		from picamera.array import PiRGBArray
		from picamera import PiCamera
		# initialize the camera and grab a reference to the raw camera capture
		camera = PiCamera()
		camera.resolution = (args.width, args.height)
		camera.framerate = 5
		time.sleep(0.1)
		rawCapture = PiRGBArray(camera, size=(args.width, args.height))
		for i in range(nbr_of_CPUs):
			p = Process(target=camThread)
			p.start()

		# capture frames from the camera
		for lframe in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
			frame = lframe.array
			rawCapture.truncate(0)
			queue.put(frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		sys.exit(0)

	elif args.oneImage != "":
		frame = cv2.imread(args.oneImage)
		if frame is None:
			raise Exception("main: Unable to load image: {}".format(args.oneImage))
		# = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		queue.put(frame)
		p = Process(target=camThread)
		p.start()
		queue.join()
		sock.shutdown(1)
		sock.close()
		sys.exit(0)

	else:	#from PC captureDevice(0 or 1)
		video_source = args.video if args.video != "" else args.captureDevice	#read video from file or webcam
		# args.captureDevice: Usually 0 will be webcam and 1 will be usb cam.
		video_capture = cv2.VideoCapture(video_source)
		video_capture.set(3, args.width)
		video_capture.set(4, args.height)
		for i in range(nbr_of_CPUs):
			p = Process(target=camThread)
			p.start()
		while True:
			ret, frame = video_capture.read()
			queue.put(frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		sys.exit(0)

# cmds:
#root@raspberrypi:/home/pi/openface/demos# ./client_pi_parallel.py --serverIP 10.40.21.38 --CPUs 4 --Gportion 1 --verbose --serverPort 12345 --threshold 0.8 --load
#root@raspberrypi:/home/pi/openface/demos# pkill -9 python2