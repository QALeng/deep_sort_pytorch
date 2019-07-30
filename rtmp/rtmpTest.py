import subprocess as sp
import cv2
import time

rtmpUrl = 'rtmp://www.iocollege.com:1935/live/mytest'
# 视频来源 地址需要替换自己的可识别文件地址
filePath='/root/qdw/deepSort/deep_sort_pytorch/'
camera = cv2.VideoCapture(filePath+"1_2s.avi") # 从文件读取视频
# 视频属性
size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
sizeStr = str(size[0]) + 'x' + str(size[1])
fps = camera.get(cv2.CAP_PROP_FPS)  # 30p/self
fps = int(fps)
hz = int(1000.0 / fps)
print('size:'+ sizeStr + ' fps:' + str(fps) + ' hz:' + str(hz))

# 视频文件输出
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(filePath+'res_mv.avi',fourcc, fps, size)
# 直播管道输出
# ffmpeg推送rtmp 重点 ： 通过管道 共享数据的方式
command = ['ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec','rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', sizeStr,
    '-r', str(fps),
    '-i', '-',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-preset', 'ultrafast',
    '-f', 'flv',
    rtmpUrl]
#管道特性配置
# pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)
pipe = sp.Popen(command, stdin=sp.PIPE) #,shell=False
# pipe.stdin.write(frame.tostring())

#业务数据计算
lineWidth = 1 + int((size[1]-400) / 400)# 400 1 800 2 1080 3
textSize = size[1] / 1000.0# 400 0.45
heightDeta = size[1] / 20 + 10# 400 20
count = 0
faces = []
while True:
    ###########################图片采集
    count = count + 1
    ret, frame = camera.read() # 逐帧采集视频流
    ############################图片输出
    # 结果帧处理 存入文件 / 推流 / ffmpeg 再处理
    pipe.stdin.write(frame.tostring())  # 存入管道用于直播
    # out.write(frame)    #同时 存入视频文件 记录直播帧数据
    # pass
camera.release()
# Release everything if job is finished
out.release()
print("Over!")
pass
