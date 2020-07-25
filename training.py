import cv2
import numpy as np
from PIL import Image
import os
import sqlite3 

def input_details(Id,fname,constt):
    connection = sqlite3.connect("details.db")
    crsr = connection.cursor() 
    crsr.execute("insert into info (id, name, num) values (?, ?, ?)",
            (Id, fname,constt))
    connection.commit() 
    connection.close() 

cam=cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


Id=input('enter your id: ')
fname=input('enter your name: ')
constt=input('enter the contact number to alert: ')
input_details(Id,fname,constt)

sampleNum=0

while(True):
	ret, img = cam.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = detector.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		#incrementing sample number
		sampleNum=sampleNum+1
		#saving the captured face in the dataset folder
		cv2.imwrite("dataSet/User."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
		cv2.imshow('frame',img)
	#wait for 100 miliseconds
	if cv2.waitKey(100) & 0xFF == ord('q'):
		break
	# break if the sample number is morethan 20
	elif sampleNum>100:
		break
cam.release()
#cv2.destroyALLWindows()

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #create empth face list
    faceSamples=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces=detector.detectMultiScale(imageNp)
        #If a face is there then append that in the list as well as Id of it
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids

faces,Ids = getImagesAndLabels('dataSet')
recognizer.train(faces, np.array(Ids))
recognizer.save('trainner/trainner.yml')