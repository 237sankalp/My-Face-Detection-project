import cv2
import pandas as pd
video = cv2.VideoCapture(0)
a=1
face_class=cv2.CascadeClassifier("C:\\Users\\Dell\\Desktop\\Anaconda\\haarcascade_frontalface_default.xml")
while True:
    a=a+1
    check,frame = video.read()
    print(check)
    print(frame)
    grey_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=face_class.detectMultiScale(grey_img,1.05,5)
    print(face)
    for x,y,w,h, in face:
        frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),3)

    cv2.imshow("Output",frame)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break

cv2.destroyAllWindows()
print(a)
video.release()