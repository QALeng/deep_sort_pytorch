from yolov3_deepsort import Detector
import numpy as np
import cv2
import time
import os
import ft2
import copy
import json
import random
import threading
import subprocess as sp


class Read_config():
    def __init__(self):
        self.configPath = 'config.json'
        self.my_config = self.readConfig(self.configPath, 'centos7')

    def readConfig(self, configPath, system):
        with open(configPath, 'r', encoding='utf-8') as f:
            all = f.read()
            temp = json.loads(all)[system]
            if temp['use_cuda'] == 'True':
                temp['use_cuda'] = True
            else:
                temp['use_cuda'] = False
            if temp['map_location_flag'] == 'True':
                temp['map_location_flag'] = True
            else:
                temp['map_location_flag'] = False
            if temp['cv2_flag'] == 'True':
                temp['cv2_flag'] = True
            else:
                temp['cv2_flag'] = False
        return temp


class play_video():
    def __init__(self):
        self.font_path = ""
        self.font_size = ""
        self.font_color = ""
        self.font_position = {}
        self.video_num = ""
        self.video_path = ""
        self.video_x_num = ""
        self.video_y_num = ""
        self.video_width = ""
        self.video_height = ""
        self.line = ""
        self.pix = ""
        self.pics = ""
        self.my_config = ""
        self.detector = []
        self.data_init()
        self.videoQueue = self.open_video()
        self.show_input = []
        self.input_arr = []
        self.output_arr = []
        self.init_video()
        # pipe初始化
        # rtmp_url='rtmp://video.510link.com:1935/live/streama'
        self.rtmp_url = 'rtmp://www.iocollege.com:1935/live/mytest'
        # 缩放比例
        self.wp, self.hp = 1, 1
        self.size = (int(self.video_width * 2 * self.wp), int(self.video_height * 2.5 * self.hp))
        self.size_str = str(self.size[0]) + 'x' + str(self.size[1])
        self.bit = '512K'
        self.fps = 12
        self.pipe = self.pipe_init(self.rtmp_url, self.size_str, self.fps, self.bit)

    # 管道初始化
    def pipe_init(self, rtmp_url, size_str, fps, bit):
        command = ['ffmpeg',
                   '-y',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-pix_fmt', 'bgr24',
                   '-s', size_str,
                   '-r', str(fps),
                   '-i', '-',
                   '-c:v', 'libx264',
                   '-pix_fmt', 'yuv420p',
                   '-preset', 'ultrafast',
                   '-rtsp_transport', 'tcp',
                   '-f', 'flv',
                   '-b:v', bit,
                   rtmp_url]
        pipe = sp.Popen(command, stdin=sp.PIPE)
        return pipe

    def data_init(self):
        config = Read_config()
        self.my_config = config.my_config
        self.font_path = "font//msyh.ttc"
        self.font_size = 18
        self.font_color = (255, 255, 255)
        self.font_position = {
            "start_x": 800,
            "start_y": 10,
            "add_x": 0,
            "add_y": 30
        }

        self.video_num = 4
        self.video_path = "video/"
        self.video_x_num = 2
        self.video_y_num = 2
        self.video_width = 500
        self.video_height = 300
        self.ft = ft2.put_chinese_text(self.font_path)
        self.detector = [Detector(self.video_width, self.video_height, self.my_config) for i in range(self.video_num)]

    def init_video(self):  # 初始化视频，防止视频打开失败
        video_path = self.video_path
        videos = os.listdir(video_path)
        for video_name in videos:
            vc = cv2.VideoCapture(video_path + video_name)  # 读入视频文件
            vc.release()
        print("初始化完毕!!")

    def showpics(self, pics, pix):  # 图片拼接
        rangx = self.video_x_num
        rangy = self.video_y_num
        vrows = []
        rows = []
        for i in range(rangy):
            numpy_vertical = []
            for j in range(rangx):

                # I just resized the image to a quarter of its original size
                #     dstheight = int(height * 0.5)  # 缩小为原来的0.5倍   可根据自己的要求定义
                #     dstwidth = int(width * 0.5)
                #     dst = cv2.resize(img, (dstwidth, dstheight), 0, 0)  # 注意width在前 height在后
                image = pics[i * 2 + j]
                if len(numpy_vertical) == 0:
                    numpy_vertical = image
                else:
                    numpy_vertical = np.hstack((numpy_vertical, image))
            if len(rows) == 0:
                rows = numpy_vertical
            else:
                rows = np.vstack((rows, numpy_vertical))
        for i in range(len(pix)):
            if len(vrows) == 0:
                vrows = pix[0]
            else:
                # 横向拼接
                vrows = np.hstack((vrows, pix[i]))
        # 纵向拼接
        rows = np.vstack((rows, vrows))
        # cv2.imshow('Numpy Vertical', rows)
        return rows

    # 打开并关闭视频流，防止视频无法打开
    def open_video(self):
        videos = os.listdir(self.video_path)
        cap = []  # 视频流数组
        for video_name in videos:

            vc = cv2.VideoCapture(self.video_path + video_name)  # 读入视频文件
            rval = vc.isOpened()
            if rval:
                cap.append(vc)
            else:
                print("error")
        return cap

    # 关闭已经打开的视频流，测试环境有效
    def close_video(self, cap):
        for vc in cap:
            vc.release()
        print("关闭视频流!!")

    # 视频帧读取子函数，依次读取视频帧
    def get_frame(self, totallen):
        rval = []
        pics = []
        img = []
        for i in range(totallen):
            # videoQueue[i].set(cv2.CAP_PROP_POS_FRAMES,index)  跳至指定帧数
            (grabbed, frame) = self.videoQueue[i].read()
            rval.append(grabbed)
            # of the stream
            if not grabbed:
                frame = np.zeros([self.video_width, self.video_height, 3], np.uint8)
            frame = cv2.resize(frame, (self.video_width, self.video_height), 0, 0)
            pics.append(frame)
        return pics

    # 视频帧读取线程函数
    def input(self):
        # 获取视频流
        totallen = len(self.videoQueue)

        while True:
            # 获取每个视频流的一帧
            pics = self.get_frame(totallen)
            # copy备份，否则被检测之后改变原有帧
            self.show_input = copy.deepcopy(pics)
            self.input_arr = pics
            # output线程结束，此线程也应该结束
            if not t3.isAlive():
                break

    # 检测线程函数
    def Handle(self):
        # 获取检测器与视频数量
        detector = self.detector
        video_num = self.video_num
        while True:
            # 获取要检测的视频帧
            pics = self.input_arr
            # 若没有，continue
            if not pics:
                continue
            # 赋值书写内容
            line = []
            linex = []
            lines = []
            pix = []
            pix_type = []
            obj_time = []
            # 检测视频
            for i in range(video_num):
                start_time = time.time()
                result = self.detector[i].detect(i + 1, pics[i], start_time)
                # result[0]=cv2.resize(result[0], (250, 150), 0, 0)
                obj_time.append(result)
            for i in range(4):
                obj_time[i][1][1] = i + random.randint(0, 9)
                obj_time[i][2][1] = random.randint(0, 9)

            for i in range(video_num):
                max = obj_time[0][1][1]
                index = 0
                for j in range(video_num - i):
                    if (obj_time[j][1][1] > max):
                        max = obj_time[j][1][1]
                        index = j
                pix.append(obj_time[index][0])
                pix_type.append(obj_time[index])
                del obj_time[index]
            pix = [cv2.resize(pix[i], (250, 150), 0, 0) for i in range(video_num)]
            self.pix = pix
            for i in range(video_num):
                linex.append(pix_type[i][1])
                linex.append(pix_type[i][2])

            for i in range(2 * video_num):
                max = linex[0][1]
                k = 0
                for j in range(2 * video_num - i):
                    try:
                        if (linex[j][1] > max):
                            max = linex[j][1]
                            k = j
                    except:
                        print(linex)
                        print(1)
                lines.append(linex[k])
                del linex[k]
            line = [lines[i][0] + ":" + str(int(lines[i][1]) // 3600) + "h" + str(
                int(lines[i][1]) % 3600 // 60) + "m" + str(int(lines[i][1]) % 60) + "s" for i in range(2 * video_num)]
            line.insert(0, '违规编号：违规时间')
            self.line = line
            # 如果output线程结束，此线程也结束
            if not t3.isAlive():
                break

    # 结果显示线程函数，包括文字书写、图片拼接与显示
    def output(self):
        # 写字位置初始化
        pos = [(self.font_position["start_x"] + i * self.font_position["add_x"],
                self.font_position["start_y"] + i * self.font_position["add_y"]) for i in range(9)]
        while True:
            # 若视频或者检测图片为空continue
            if not self.show_input:
                continue
            if not self.pix:
                continue
            # start=time.time()    #FPS  start_time
            # 获取书写内容
            line = self.line
            # 拼接四张视频和四张图片
            allframe = self.showpics(self.show_input, self.pix)
            # 书写
            for i in range(len(line)):
                allframe = self.ft.draw_text(allframe, pos[i], line[i], self.font_size, self.font_color)

            # 保存改造好的图片，为视频传输做准备。
            self.pics = allframe

            # 显示
            cv2.namedWindow('结果', cv2.WINDOW_NORMAL)
            # 窗体全屏化
            cv2.setWindowProperty('结果', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('结果', allframe)
            # end=time.time()       #FPS  end_time
            # print("FPS:",1/(end-start))
            # 若检测到esc,则退出循环结束线程，esc的ascll码是27
            if (cv2.waitKey(1) == 27):
                break

    def delivery(self):
        pic = self.pics
        pos = [(self.font_position["start_x"] + i * self.font_position["add_x"],
                self.font_position["start_y"] + i * self.font_position["add_y"]) for i in range(9)]
        while True:
            if not self.show_input:
                continue
            if not self.pix:
                continue
            # start=time.time()    #FPS  start_time
            # 获取书写内容
            line = self.line
            # 拼接四张视频和四张图片
            allframe = self.showpics(self.show_input, self.pix)
            # 书写
            for i in range(len(line)):
                allframe = self.ft.draw_text(allframe, pos[i], line[i], self.font_size, self.font_color)

            # 保存改造好的图片，为视频传输做准备。
            self.pics = allframe
            # 视频传输代码

            self.pipe.stdin.write(self.pics.tostring())
            time.sleep(1)
            # 如果output线程结束，此线程也结束
            if not t3.isAlive():
                break


if __name__ == '__main__':
    video = play_video()
    # 线程一：从rtmp流中读取帧送入输入队列中
    t1 = threading.Thread(target=video.input, name="Job1", args=())
    # 线程二：yolov3训练并输出结果保存至输出队列中
    t2 = threading.Thread(target=video.Handle, name="Job2", args=())
    # 线程三：显示模块不停读取输出队列，若有则显示，若无则continue
    # t3 = threading.Thread(target=video.output, name="Job3", args=())
    # 线程四：视频传输模块
    t3 = threading.Thread(target=video.delivery, name="Job4", args=())

    t1.start()
    t2.start()
    t3.start()
    # t4.start()









