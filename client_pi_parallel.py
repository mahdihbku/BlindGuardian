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
from random import randint
from multiprocessing import Pool
from multiprocessing import Process, JoinableQueue, Lock, Manager

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor',	type=str, help="Path to dlib's face predictor.", default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel',		type=str, help="Path to Torch network model.", default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim',			type=int, help="Default image dimension.", default=96)
parser.add_argument('--serverPort',		type=int, help="Port of the server.", default=6546)
parser.add_argument('--serverIP',		type=str, help="IP address of the server.", default="127.0.0.1")
parser.add_argument('--threshold',		type=float, default=0.99)
parser.add_argument('--verbose',		action='store_true')
parser.add_argument('--captureDevice',		type=int, default=0, help='Capture device. 0 for latop webcam and 1 for usb webcam')
parser.add_argument('--width',			type=int, default=640)
parser.add_argument('--height',			type=int, default=480)
parser.add_argument('--load',			action='store_true')
parser.add_argument('--CPUs',			type=int, help="Number of parallel CPUs to be used.", default=4)
parser.add_argument('--Gportion',		type=int, help="Portion of G to be stored locally (0: no G, 1:full G, 2:half G, 3:third G, 4:quarter G.", default=1)
parser.add_argument('--maxRam',			type=int, help="Maximum amount of ram to be used by the system (in GB).", default=5)
args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
#net = openface.TorchNeuralNet(args.networkModel, args.imgDim)

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
#transition_time = 2	# (seconds)
nbr_of_CPUs = args.CPUs
max_ram = args.maxRam
G_portion = args.Gportion

# Temporary global variables
B = []
G = []
#F = []
C = []
r1_list = []	#list of randomly generated r1
r2_list = []	#list of randomly generated r2
#imgrep = []
queue = JoinableQueue(maxsize=nbr_of_CPUs)
lock = Lock()			# mutex to protect persons_reps
manager = Manager()
persons_reps = manager.list()	# detected faces by the camera

def getRep(bgrImg, net):
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
	# Create a TCP/IP socket
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	# Connect the socket to the port where the server is listening
	server_address = (server_ip, server_port)
	print >>sys.stderr, 'connecting to %s port %s' % server_address
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
		print 'getPubKey:error'

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
		print 'getBCfiles:error'

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
	# ec_elgamal.prepare(pub_key_file, "priv.txt")

	pool = Pool(processes=nbr_of_CPUs)
	global G_portion
	G = []
	C = np.load(C_file)
	suspects_count = len(C)
	#if verbose:	print("generateLocalFiles:suspects_count={}".format(suspects_count))
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

	if verbose:	print("generateLocalFiles:local files have been generated in {}".format((end_gen_files-start_gen_files)*1000))

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
	for x in normalizedRep:
		if x>255:
			x = 255
		elif x<0:
			x = 0
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
	#t_start = time.time()
	enc_y2 = ec_elgamal.encrypt_ec(str(sum([y**2 for y in imgrep])))
	#t_end = time.time()
	#print("compute_sub_scores: encrypting from sum_y_j={} y_j = {} ms".format(sum([y**2 for y in imgrep]), (t_end-t_start)*1000))
	
	#print("generateLocalFiles:len(F)={}".format(len(F)))


	#t_start = time.time()
	#enc_d = F[imgrep[0]]
	#for j in range(1, 128):
		#enc_d = ec_elgamal.add2(enc_d, F[imgrep[j]])		# check which one is faster (128 adds or 1 enc). checked!
	#t_end = time.time()
	#print("compute_sub_scores: encrypting from F[] = {} ms".format((t_end-t_start)*1000))

	if G_portion==0:
		for i in list:
			enc_d = ec_elgamal.add2(enc_y2, B[i])	# B_i = sum_j(x_ij^2)
			Ci = C[i]
			for j in range(0, 128):
				enc_d = ec_elgamal.add2(enc_d, ec_elgamal.mult(str(imgrep[j]), Ci[j]))
			D.append(enc_d)
	elif G_portion==1:
		for i in list:
			enc_d = ec_elgamal.add2(enc_y2, B[i])	# B_i = sum_j(x_ij^2)
			Gi = G[i]
			for j in range(0, 128):
				enc_d = ec_elgamal.add2(enc_d, Gi[j][imgrep[j]])
			D.append(enc_d)
	else:
		first_index = 64 if G_portion==2 else 85 if G_portion==3 else 96 if G_portion==4 else 0
		last_index = 192 if G_portion==2 else 171 if G_portion==3 else 160 if G_portion==4 else 0
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
	print("thread started...")
	net = openface.TorchNeuralNet(os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'), 96)
	while True:
		frame = queue.get()
		start_comp_time = time.time()
		face_reco_start_time = time.time()
		if frame is None:
			break
		repsAndBBs = getRep(frame, net)
		reps = repsAndBBs[0]
		if (len(reps) == 0):	# no face in frame
			print("camThread:no face in the frame")
			queue.task_done()
			continue
		bbs = repsAndBBs[1]
		reps_array = np.array([normalizeRep(face) for face in reps])
		person_id = -1
		lock.acquire()
		#print("camThread:lock.acquire(), len(persons_reps)={}".format(len(persons_reps)))
		for idx, rep in enumerate(persons_reps):
			dist = LA.norm(reps_array-rep)
			#print("camThread:dist={} similarity_threshold*normalizing_multiplier={}".format(dist, similarity_threshold*normalizing_multiplier))
			if dist < similarity_threshold*normalizing_multiplier :
				person_id = idx
				lock.release()
				print("camThread:no new face in the frame")
				#print("camThread:lock.release(), len(persons_reps)={}".format(len(persons_reps)))
				break
		if person_id == -1:		# if a new person has been detected
			print("camThread:a new face in the frame")
			face_reco_end_time = time.time()
			if verbose:	print("main:face recognition = {} ms".format((face_reco_end_time-face_reco_start_time)*1000))
			start_enc_time = time.time()
			persons_reps.append(reps_array)
			person_id += 1
			lock.release()
			#print("camThread:lock.release(), len(persons_reps)={}".format(len(persons_reps)))
			imgrep = reps_array[0]


			#print("camThread:reps={}".format(reps))
			#print("camThread:reps_array={}".format(reps_array))
			#print("camThread:len(imgrep)={}".format(len(imgrep)))
			#print("camThread:imgrep={}".format(imgrep))
			#print("camThread:imgrep[0]={}".format(imgrep[0]))


			dist_comp_start = time.time()
			#pool = Pool(processes=nbr_of_CPUs)
			#suspects_indices = range(suspects_count)
			#D = pool.map(compute_sub_scores, (suspects_indices[int(i*suspects_count/nbr_of_CPUs):int((i+1)*suspects_count/nbr_of_CPUs)] for i in range(nbr_of_CPUs)))
			#D = [ent for sublist in D for ent in sublist]
			#pool.close()
			D = compute_sub_scores(range(suspects_count), imgrep)
			dist_comp_end = time.time()
			if verbose:	print("main:computing the distance = {} ms".format((dist_comp_end-dist_comp_start)*1000))


			#print("decrypting D...")
			#print("decrypted_D_0 = {}".format(ec_elgamal.decrypt_ec(D[0])))
			#print("imgrep={}".format(imgrep))
			

			dist_obf_start = time.time()
			f = open(rand_numbers_file, "rb")	#TODO read the file once at the begining
			D = [ec_elgamal.add2(ec_elgamal.mult(r1_list[i], ec_elgamal.add2(D[i], enc_similarity_threshold)), r2_list[i]) for i in range(suspects_count)]
			#D = [ec_elgamal.add2(D[i], enc_similarity_threshold) for i in range(suspects_count)]	#Without obfuscation
			f.close()
			dist_obf_end = time.time()
			if verbose:	print("main:obfuscating the distance = {} ms".format((dist_obf_end-dist_obf_start)*1000))

			dist_comp_obf_end = time.time()
			if verbose:	print("main:time(face recognition + scores computation + scores obfuscation) for {} suspects: {} ms".format(suspects_count, (dist_comp_obf_end-start_comp_time)*1000))
			if verbose:	print("main:enc time for {} suspects: {} ms".format(suspects_count, (dist_comp_obf_end-start_enc_time)*1000))

			results_file = open("final_results.txt","a+")
			results_file.write("Online:M= {} CPUs_camera= {} dist_comp= {} dist_obf= {} total_time= {} enc_time= {}\n".format(suspects_count,nbr_of_CPUs,(dist_comp_end-dist_comp_start)*1000,(dist_obf_end-dist_obf_start)*1000,(dist_comp_obf_end-start_comp_time)*1000,(dist_comp_obf_end-start_enc_time)*1000))
			results_file.close()

			enc_D = pickle.dumps(D)
			sendScores(sock, enc_D)
			print("main:The encrypted scores have been sent to the server")
			data = sock.recv(11)
			if (data == "GET image  "):
				# sending the suspects image
				# TODO blure irrelevant faces in the frame
				image = pickle.dumps(frame)
				send_msg(sock, image)
				print("main:Suspect detected! The image of the suspect has been sent")
				end_comp_time = time.time()
				if verbose:	print("main:suspect detected: total rtt: {} ms".format((end_comp_time-start_comp_time)*1000))
				
				results_file = open("final_results.txt","a+")
				results_file.write("Online:RTT= {}\n".format(end_comp_time-start_comp_time))
				results_file.close()


if __name__ == '__main__':
	global G_portion
	# Offline phase___________________________________________
	sock = connectToServer()

	if not load:
		if verbose:	print("main:getting remote files...")
		getPubKey(sock)
		if verbose:	print("main:pub_key received and saved in {}".format(pub_key_file))
		getBCfiles(sock)
		if verbose:	print("main:server files received and saved in {}, {}".format(B_file, C_file))
		if verbose:	print("main:generating local files (G_portion={})...".format(G_portion))
		generateLocalFiles()
		if verbose:	print("main:local files have been generated and saved in {}".format(G_file))
		if verbose:	print("main:generating random numbers file...")
		fill_rand_numbers_file()
		if verbose:	print("main:random files file {} has been filled".format(rand_numbers_file))

	# Online phase____________________________________________
	ec_elgamal.prepare_for_enc(pub_key_file)
	#ec_elgamal.prepare(pub_key_file, "ec_priv.txt") #for debugging only TODO to remove

	enc_similarity_threshold = ec_elgamal.encrypt_ec(str(-(similarity_threshold*normalizing_multiplier)**2))

	B = np.load(B_file)
	suspects_count = len(B)
	F = np.load(str(F_file+".npy"))
	f = open(rand_numbers_file, "rb")
	r1 = f.readline()
	while r1:
		r1_list.append(r1)
		r2 = f.read(ec_elgamal_ct_size)
		r2_list.append(r2)
		r1 = f.readline()
		if not r1:
			break
	f.close()
	
	
	print("generateLocalFiles:len(F)={}".format(len(F)))

	if G_portion != 1:
		C = np.load(C_file)
	if G_portion != 0:
		G = np.load(str(G_file+".npy"))
		#Get G_portion from the stored G
		G_portion = 1 if len(G[0][0])==256 else 2 if len(G[0][0])==128 else 3 if len(G[0][0])==86 else 4 if len(G[0][0])==64 else 0
	print("main:G_portion={}".format(G_portion))

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

# cmds:
#root@raspberrypi:/home/pi/openface/demos# ./client_pi_parallel.py --serverIP 10.40.21.38 --CPUs 4 --Gportion 1 --verbose --serverPort 12345 --threshold 0.8 --load
#root@raspberrypi:/home/pi/openface/demos# pkill -9 python2