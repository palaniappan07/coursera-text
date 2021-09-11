import cv2
from random import randrange

# load some pre trained data on face frontals
trained_face_data=cv2.CascadeClassifier(r'/Users/palaniappanarunachalam/PycharmProjects/face detection/venv/haarcascade_frontalface_default.xml')

#choose an image
img =cv2.imread(r'/Users/palaniappanarunachalam/PycharmProjects/face detection/venv/_UMA4251.JPG')
#convert to grayscale
gs_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect Faces
face_coordinates = trained_face_data.detectMultiScale(gs_img)
#print(face_coordinates)
#draw rectangle
for (x,y,w,h) in face_coordinates:
   cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)




#show
cv2.imshow('clever programmer Face', img)
cv2.waitKey()
print("code completed")