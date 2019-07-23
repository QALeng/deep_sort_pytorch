from demo_yolo3_deepsort import Detector
import  time
import  cv2

width=640
height=360
detector=Detector(width,height)
video_path='./video/2.mp4'
cv2.namedWindow("test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("test", 800, 600)
vdo=cv2.VideoCapture()
stream=vdo.open(video_path)
while vdo.grab():
    start_time = time.time()
    print('now_time: ',start_time)
    _, ori_im = vdo.retrieve()
    result1,result2=detector.detect(ori_im,start_time)
    # print(result1)
    cv2.imshow("test", result1)
    cv2.waitKey(1)