import cv2
import numpy as np

# Import the model
# import tensorflow
from tensorflow.keras.models import load_model
model = load_model('tomcruise_detector.model')

face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# face_clsfr = cv2.CascadeClassifier(cascPath)

# Capturing the frames
input_feed = cv2.VideoCapture(0)

labels_dict = {0:'Normal Human',1:'Tom Cruise'}
color_dict = {0:(0,0,255),1:(0,255,0)}


while(True):

    # Get individual frame
    ret, img = input_feed.read()
    if(img is not None):
        # converting to grey scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_clsfr.detectMultiScale(gray,1.3,5)  

    for (x,y,w,h) in faces:
        # normalizing and resizing ... further predicting
        face_img = gray[y:y+w,x:x+w]
        resized = cv2.resize(face_img,(100,100))
        normalized = resized/255.0
        reshaped = np.reshape(normalized,(1,100,100,1))
        result = model.predict(reshaped)

        label = np.argmax(result,axis=1)[0]
        
        # dimensions of rectangle around image
        cv2.rectangle(img,(x,y), (x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y), (x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)
    key = cv2.waitKey(1)
    
    # escape key to come out of process!
    if(key == 27):
        break
        
cv2.destroyAllWindows()
input_feed.release()