import os, cv2
import numpy as np 
from pygame import mixer
from keras.models import load_model

mixer.init()
sound = mixer.Sound('alarm.wav')

# using haar cascade files we can detect specific features of a face
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
eye = cv2.CascadeClassifier('haar cascade files\haarcascade_eye.xml')
l_eye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
r_eye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

labels = ['Closed', 'Open']
model = load_model('models/cnn_2.h5')

path = os.getcwd()          # gets cur working dir
cap = cv2.VideoCapture(0)   # uses the camera
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thickness = 0
rpred = lpred = [99]

while(True):
    ret, frame = cap.read()         #Capture frame-by-frame
    height, width = frame.shape[:2]     #Specifies size of frame

    if not ret:  # Check if frame retrieval was successful
        print("Error: Unable to retrieve frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      #Convert to grayscale
    
    faces = face.detectMultiScale(gray, minNeighbors=5, minSize=(25,25), scaleFactor=1.1)       #detects features of diff sizes
    eyes = eye.detectMultiScale(gray)
    leftEye = l_eye.detectMultiScale(gray)
    rightEye = r_eye.detectMultiScale(gray)

    #cv2.rectangle(image, start point, end point, color, thickness)
    cv2.rectangle(frame, (0, height-50), (200, height), (0,0,0), thickness=cv2.FILLED)           #creates a rectangle around the feature

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (150,150,150) , 1) 
        
    for (x,y,w,h) in eyes:
        cv2.rectangle(frame, (x,y),(x+w,y+h) ,(150,150,150) , 1) 

    for(x,y,w,h) in rightEye:
        rEye = frame[y:y+h, x:x+w]
        count = count + 1
        # convert eye feature to grayscale, resize pixels, normalize pixel values
        rEye = cv2.cvtColor(rEye, cv2.COLOR_BGR2GRAY)
        rEye = cv2.resize(rEye, (100,100))
        rEye = rEye/255
        rEye = rEye.reshape(100,100,-1)

        # reshaping and expanding dims to match CNN inputs
        rEye = np.expand_dims(rEye, axis=0)
        rpred = model.predict(rEye)
        
        if rpred < 0.5:
            labels = 'Closed'
        else:
            labels = "Open"
        break

    for(x,y,w,h) in leftEye:
        lEye = frame[y:y+h, x:x+w]
        count = count + 1
        # convert eye feature to grayscale, resize pixels, normalize pixel values
        lEye = cv2.cvtColor(lEye, cv2.COLOR_BGR2GRAY)
        lEye = cv2.resize(lEye, (100,100))
        lEye = lEye/255
        lEye = lEye.reshape(100,100,-1)

        # reshaping and expanding dims to match CNN inputs
        lEye = np.expand_dims(lEye, axis=0)
        lpred = model.predict(lEye)

        if lpred < 0.5:
            labels = 'Closed'
        else:
            labels = "Open"
        break

    if(rpred < 0.5 and lpred < 0.5):
        score=score + 1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    else:
        score=score - 1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score > 15):
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
        except:
            pass
        if(thickness < 16):
            thickness = thickness + 2
        else:
            thickness = thickness - 2
            if(thickness < 2):
                thickness = 2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thickness)
    cv2.imshow("Frame", frame)          # show enhanced frame
    if cv2.waitKey(1) & 0xFF == ord('z'): break
cap.release()
cv2.destroyAllWindows()
        

