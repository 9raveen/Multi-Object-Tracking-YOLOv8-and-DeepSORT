import cv2
cap = cv2.VideoCapture('outputs/output_video.mp4')
print('Opened:', cap.isOpened())
print('FPS:', cap.get(cv2.CAP_PROP_FPS))
print('Frame count:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
ret, frame = cap.read()
print('Read frame:', ret)
if ret:
    cv2.imwrite('outputs/frame_preview.jpg', frame)
cap.release()
