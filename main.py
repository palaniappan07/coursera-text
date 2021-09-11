import cv2
from random import randrange

#ai
trained_face_detection=cv2.CascadeClassifier('/Users/palaniappanarunachalam/PycharmProjects/webcam detection/venv/haarcascade_frontalface_default.xml')
#to capture video from webcam
webcam=cv2.VideoCapture(0)
key=cv2.waitKey(1)

#iterate forever over the frames
while True:
    # read the current frame
    successful_frame_read, frame=webcam.read()

    #convert to grayscale
    gs_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect Faces
    face_coordinates = trained_face_detection.detectMultiScale(gs_img)

    # draw rectangle
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)

    cv2.imshow("output",frame)
    key=cv2.waitKey(1)

    #####stop if q is pressed
    if key==81 or key==113:
        break


###release video capture
webcam.release()



print("code completed")




