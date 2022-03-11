import cv2
import numpy as np
def face_detect_demo(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_detector = cv2.CascadeClassifier('/Users/maojietang/opt/anaconda3/envs/tf/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    face = face_detector.detectMultiScale(gray)
    for x,y,w,h in face:
        cv2.rectangle(img, (x,y),(x+w, y+h),color=(0,0,255), thickness=2)
    cv2.imshow('Result', img)

# img = cv2.imread('/Users/maojietang/Downloads/Face.jpg')
cap = cv2.VideoCapture('/Users/maojietang/Desktop/test1.mov')
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
cap.read()

while True:
    flag, frame = cap.read()
    if not flag:
        break
    face_detect_demo(frame)
    if ord('q') == cv2.waitKey(1):
        break
cv2.destroyAllWindows()
cap.release()