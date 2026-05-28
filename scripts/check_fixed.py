import cv2
cap=cv2.VideoCapture('outputs/output_fixed.mp4')
print('Opened',cap.isOpened())
print('Frames', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
ret,frame=cap.read()
print('Read',ret)
if ret:
    cv2.imwrite('outputs/frame_preview_fixed.jpg', frame)
cap.release()
