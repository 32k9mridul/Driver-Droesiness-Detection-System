from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import os
import numpy as np
from PIL import Image
import geocoder
from twilio.rest import Client
import sqlite3 

def fetch_name(Id):
	connection = sqlite3.connect("details.db")
	crsr = connection.cursor()
	for row in crsr.execute("SELECT name FROM info where id=Id"):
		return row[0]

def fetch_number(Id):
	connection = sqlite3.connect("details.db")
	crsr = connection.cursor()
	for row in crsr.execute("SELECT num FROM info where id=Id"):
		return row[0]


def get_current_location():
    g_maps_url = "http://maps.google.com/?q={},{}"
    g = geocoder.ip('me')
    #lat = g.latlng[0] + 2.64
    lat = g.latlng[0]
    #long = g.latlng[1] + 1.3424
    long = g.latlng[1]
    #print(lat, long)
    current_location =  g_maps_url.format(lat, long)
    return current_location

def send_alert(current_location,tmes):
    # Your Account SID from twilio.com/console  
    account_sid = "AC07949d8429142271c08317a702d349bf"
    # Your Auth Token from twilio.com/console
    auth_token  = "9e6d46f60ddd562ea48a227b61128455"

    client = Client(account_sid, auth_token)
    temp="+91"+str(contt)
    message = client.messages.create(
    to=temp, 
    from_="+19143038893",
    body=tmes+current_location)

    print("message sent to owner\n")

def sound_alarm(path):
	current_location=get_current_location();
	send_alert(current_location,"\nDriver is Droswy\n")
	playsound.playsound(path)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])


    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def mouth_aspect_ratio(mou):
	# compute the euclidean distances between the horizontal
	X   = dist.euclidean(mou[0], mou[6])
	# compute the euclidean distances between the vertical
	Y1  = dist.euclidean(mou[2], mou[10])
	Y2  = dist.euclidean(mou[4], mou[8])
	# taking average
	Y   = (Y1+Y2)/2.0
	# compute mouth aspect ratio
	mar = Y/X
	return mar


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

cam = cv2.VideoCapture(0)
#font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)

fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)
cnt=0
cnt2=0
contt=8800944390
fname="Unknown"
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf<40):
            Id=fetch_name(Id)
            contt=fetch_number(Id)
            fname=fetch_name(Id)
        else:
            Id="Unknown"
            cnt2=cnt2+1
            if(cnt2==10):
            	current_location=get_current_location()
            	mes="\nunkown autherizaton\n"
            	send_alert(current_location,mes)
            	print("\n Unknown autherizaton\n")
        cv2.putText(im, str(Id), (x,y+h), fontface, fontscale, fontcolor)
        #cv2.cv.PutText(cv2.cv.fromarray(im),str(Id), (x,y+h),font, 255)
    cv2.imshow('im',im) 
    cnt=cnt+1
    if ((cv2.waitKey(10) & 0xFF==ord('q')) | cnt==20):
        break
cam.release()
#cv2.destroyAllWindows()

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 30
MOU_AR_THRESH = 0.75

COUNTER = 0
ALARM_ON = False
yawnStatus = False
yawns = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print("\nAUTHENTICATION COMPLETE\n")
print("Welcome"+" "+fname)
print("\n starting video stream...\n")
vs = VideoStream(0).start()
time.sleep(1.0)

while True:
	
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	prev_yawn_status = yawnStatus
	rects = detector(gray, 0)

	
	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)


		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		mouth = shape[mStart:mEnd]

		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		mouEAR = mouth_aspect_ratio(mouth)
		
		ear = (leftEAR + rightEAR) / 2.0

		
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
		
		if ear < EYE_AR_THRESH:
			COUNTER += 1

			
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				
				if not ALARM_ON:
					ALARM_ON = True

					
					if ('alarm.WAV') != "":
						t = Thread(target=sound_alarm('alarm.WAV'),
							args=('alarm.WAV'))
						t.deamon = True
						t.start()

				
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		
		else:
			COUNTER = 0
			ALARM_ON = False

		
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
		if mouEAR> MOU_AR_THRESH:
			cv2.putText(frame, "Yawning ", (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			yawnStatus=True
			output_text="Yawn Count: " + str(yawns + 1)
			cv2.putText(frame, output_text, (10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)
		else:
			yawnStatus=False
		if prev_yawn_status == True and yawnStatus == False:
			yawns+=1
		cv2.putText(frame, "MAR: {:.2f}".format(mouEAR), (480, 60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 	
	
	if key == ord("q"):
		break


cv2.destroyAllWindows()
vs.stop()