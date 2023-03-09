import cv2

myrtmp_addr = "rtmp://172.17.0.2:1935/live/test"
cap = cv2.VideoCapture(myrtmp_addr)

print(cap.read())




