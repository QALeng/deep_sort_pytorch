# -*- coding:utf-8 -*-
from demo_yolo3_deepsort import Detector
from rtmp.rtmpFunC import pipeFuncInit
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
my_config=readConfig(configPath,'centos7')
print(my_config)
width=640
height=360

detector=Detector(width,height,my_config)
video_path='./video/2.mp4'
# cv2.namedWindow("test", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("test", 800, 600)
vdo=cv2.VideoCapture()
# stream=vdo.open(video_path)

stream=cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output = cv2.VideoWriter("1_60s.avi", fourcc, 20, (width, height))
#pipe
rtmpUrl = 'rtmp://www.iocollege.com:1935/live/mytest'
newWidth=stream.get(cv2.CAP_PROP_FRAME_WIDTH)
newHeight=stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
sizeStr=str(int(newWidth))+'x'+str(int(newHeight))
fps=stream.get(cv2.CAP_PROP_FPS)
stream.release()
stream=vdo.open(video_path)
pipe=pipeFuncInit(rtmpUrl, sizeStr, fps)

while vdo.grab():
    start_time = time.time()
    print('now_time: ',start_time)
    _, ori_im = vdo.retrieve()
    result1,result2=detector.detect(1,ori_im,start_time)
    # print(result1)
    # cv2.imshow("test", result1)
    output.write(result1)

    pipe.stdin.write(result1.tostring())  # 存入管道用于直播
    # cv2.waitKey(1)
stream.release()
output.release()