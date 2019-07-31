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
        translate_dict={'True':True,"False":False}
        keys=['use_cuda','map_location_flag','cv2_flag']
        for i in keys:
            temp[i]=translate_dict[temp[i]]
    return temp


#配置文件
configPath='config.json'
my_config=read_config(configPath,'centos7')
#视频名称
video_name='14.mp4'
#帧的宽，高度
width,height=1920,1080
#自行设置
fps=24
#检测函数初始化
detector=Detector(width,height,my_config)


#输入视频
video_dir='./video/'
vdo=cv2.VideoCapture()
vdo.open(video_dir+video_name)


#pipe初始化
# rtmp_url='rtmp://video.510link.com:1935/live/streama'
rtmp_url = 'rtmp://www.iocollege.com:1935/live/mytest'
#缩放比例
wp,hp=1,1
size=(int(width*wp),int(height*hp))
size_str=str(size[0])+'x'+str(size[1])
bit='512K'
pipe=pipe_init(rtmp_url,size_str , fps,bit)


#输出文件
#输入文件的目录
video_save_dir='./video_pre/'
# 用来设置需要保存视频的编码方式
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
save_name=video_name.split('.')[0]+'.avi'
output = cv2.VideoWriter(video_save_dir+save_name, fourcc, fps, (width, height))


while vdo.grab():
    start_time = time.time()
    _, ori_im = vdo.retrieve()
    ori_im,return_time=detector.detect(1,ori_im,start_time)
    output.write(ori_im)
    dispose_img=cv2.resize(ori_im,size)
    pipe.stdin.write(dispose_img.tostring())  # 存入管道用于直播


vdo.release()