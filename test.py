# -*- coding:utf-8 -*-
from yolov3_deepsort import Detector
from rtmp.rtmp_func import pipe_init
import  time
import  cv2
import json
import copy

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
configPath='config/config.json'
my_config=read_config(configPath,'win10')
#视频名称
video_name='2.mp4'
#帧的宽，高度
# width,height=1920,1080
# width,height=976,504
width,height=640,360
#自行设置
fps=24
#检测函数初始化
detector=Detector(width,height,my_config)


#输入视频
video_dir='./video/'
vdo=cv2.VideoCapture()
vdo.open(video_dir+video_name)


# #pipe初始化
# # rtmp_url='rtmp://video.510link.com:1935/live/streama'
# rtmp_url = 'rtmp://www.iocollege.com:1935/live/mytest'
# #缩放比例
# wp,hp=1,1
# size=(int(width*wp),int(height*hp))
# size_str=str(size[0])+'x'+str(size[1])
# bit='512K'
# # pipe=pipe_init(rtmp_url,size_str , fps,bit)


#输出文件
#输入文件的目录
video_save_dir='./video_pre/'
# 用来设置需要保存视频的编码方式
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
save_name='yolov3_'+video_name.split('.')[0]+'.avi'
output = cv2.VideoWriter(video_save_dir+save_name, fourcc, fps, (width, height))


class DectetData():
    def __init__(self):
        self.index=0
        self.ori_im=[]
        self.start_time=[]
new_data=DectetData()


cv2.namedWindow("test",cv2.WINDOW_NORMAL)
cv2.resizeWindow("test",640,360)
while vdo.grab():
    start_time = time.time()
    _, ori_im = vdo.retrieve()
    new_data.index=0
    new_data.ori_im=copy.deepcopy(ori_im)
    new_data.start_time=start_time
    return_data=detector.detect([new_data])

    if(len(return_data)==0):
        continue
    print("test")
    print(type(return_data[0][1]))
    cv2.imshow('test',return_data[0][1])
    # output.write(new_data.ori_im)
    # dispose_img=cv2.resize(ori_im,size)
    # pipe.stdin.write(dispose_img.tostring())  # 存入管道用于直播
    cv2.waitKey(1)

vdo.release()