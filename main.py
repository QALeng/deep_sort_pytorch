# -*- coding:utf-8 -*-
from demo_yolo3_deepsort import Detector
import  time
import  cv2
import json

def readConfig(configPath,system):
    with open(configPath, 'r', encoding='utf-8') as f:
        all = f.read()
        temp = json.loads(all)[system]
        if temp['use_cuda'] == 'True':
            temp['use_cuda'] = True
        else:
            temp['use_cuda'] = False
        if temp['map_location_flag']=='True':
            temp['map_location_flag']=True
        else:
            temp['map_location_flag']=False
        if temp['cv2_flag'] == 'True':
            temp['cv2_flag'] = True
        else:
            temp['cv2_flag'] = False
    return temp


configPath='config.json'
my_config=readConfig(configPath,'win10')
print(my_config)
width=640
height=360

detector=Detector(width,height,my_config)
video_path='./video/2.mp4'
cv2.namedWindow("test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("test", 800, 600)
vdo=cv2.VideoCapture()
stream=vdo.open(video_path)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output = cv2.VideoWriter("1_60s.avi", fourcc, 20, (width, height))
while vdo.grab():
    start_time = time.time()
    print('now_time: ',start_time)
    _, ori_im = vdo.retrieve()
    result1,result2=detector.detect(1,ori_im,start_time)
    # print(result1)
    cv2.imshow("test", result1)
    output.write(result1)
    cv2.waitKey(1)