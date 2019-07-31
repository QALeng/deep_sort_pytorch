# -*- coding:utf-8 -*-
from yolov3_deepsort import Detector
from rtmp.rtmp_func import pipe_init
import  time
import  cv2
import json

#读取配置文件函数
def read_config(configPath,system):
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

#配置文件
configPath='config.json'
my_config=read_config(configPath,'centos7')
#帧的宽，高度
width,height=1280,720
#检测函数初始化
detector=Detector(width,height,my_config)
video_name='11.mp4'
video_path='./video/'+video_name
vdo=cv2.VideoCapture()
stream=cv2.VideoCapture(video_path)




#pipe初始化
# rtmp_url = 'rtmp://www.iocollege.com:1935/live/mytest'
rtm_url='rtmp://video.510link.com:1935/live/streama'
wp,hp=0.5,0.5
size=(int(width*wp),int(height*hp))
size_str=str(size[0])+'x'+str(size[1])
fps=24
bit='512K'
pipe=pipe_init(rtm_url,size_str , fps,bit)


#输出文件
# 用来设置需要保存视频的格式
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output = cv2.VideoWriter("./video_pre/"+video_name, fourcc, fps, (width, height))



while vdo.grab():
    start_time = time.time()
    _, ori_im = vdo.retrieve()
    ori_im,return_time=detector.detect(1,ori_im,start_time)
    pipe.stdin.write(ori_im.tostring())  # 存入管道用于直播
    output.write(ori_im)

stream.release()