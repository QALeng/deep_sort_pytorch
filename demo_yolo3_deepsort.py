import os
import cv2
import numpy as np

from YOLO3 import YOLO3
from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes

from config import my_config

import time


class Detector(object):
    def __init__(self):
        self.vdo = cv2.VideoCapture()
        self.yolo3 = YOLO3("YOLO3/cfg/yolo_v3.cfg", "YOLO3/yolov3.weights", "YOLO3/cfg/coco.names", is_xywh=True)
        self.deepsort = DeepSort("deep/checkpoint/ckpt.t7")
        self.class_names = self.yolo3.class_names
        self.write_video = True
        self.record_object={}  #
        self.need=my_config['need']


    def open(self, video_path):
        assert os.path.isfile(video_path), "Error: path error"
        self.vdo.open(video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.area = 0, 0, self.im_width, self.im_height
        if self.write_video:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter("demo2222.avi", fourcc, 20, (self.im_width, self.im_height))
        return self.vdo.isOpened()

    def detect(self):
        # batch_size=8
        xmin, ymin, xmax, ymax = self.area
        class_name = self.class_names
        record_object = self.record_object
        need=self.need
        #开始时间先模拟
        start_time=time.time()
        while self.vdo.grab():
            start = time.time()
            start_time = time.time()
            print('now_time: ',start_time)
            endTime = start
            # 方法/函数解码并返回刚刚抓取的帧。如果没有抓取帧（摄像机已断开连接，或视频文件中没有帧），则方法返回false，函数返回NULL指针。
            # https://docs.opencv.org/3.1.0/d8/dfe/classcv_1_1VideoCapture.html#a9ac7f4b1cdfe624663478568486e6712
            # 一帧一帧的读取

            _, ori_im = self.vdo.retrieve()

            im = ori_im[ymin:ymax, xmin:xmax, (2, 1, 0)]
            # 用于检测物体
            bbox_xywh, cls_conf, cls_ids = self.yolo3(im)


            if bbox_xywh is not None:
                # mask = cls_ids == 1
                # 用于在图片上画框
                mask = [i in need for i in cls_ids]
                all_name = [class_name[int(i)] for i in cls_ids if i in need]
                bbox_xywh = bbox_xywh[mask]
                bbox_xywh[:, 3] *= 1.2
                cls_conf = cls_conf[mask]
                outputs, total_name,stay_time = self.deepsort.update(bbox_xywh, cls_conf, im, all_name,start_time)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]  # 可以通过这里来记录时间，因为这里可以查看当前对象的id
                    length = len(total_name)
                    # 添加新的
                    for i in range(length):
                        objectId = identities[i]
                        objectName = total_name[i]
                        if (objectId not in record_object.keys()):
                            record_object[objectId] = [objectName, start]
                    keys_arr = record_object.keys()
                    # print(keysArr)
                    end = [[one, endTime] for one in keys_arr if one not in identities]
                    endObject = [[endOne[0], record_object[endOne[0]] + [endTime]] for endOne in end]
                    # print(endObject)
                    for endOne in end:
                        del record_object[endOne[0]]
                    # print(recordObject)
                    new_ori_im= draw_bboxes(ori_im, bbox_xyxy, identities, total_name, offset=(xmin, ymin))
                print(stay_time)
            if (cv2_flag == True):
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)


            if self.write_video:
                self.output.write(ori_im)

        return

if __name__ == "__main__":
    # import sys
    # if len(sys.argv) == 1:
    #     print("Usage: python demo_yolo3_deepsort.py [YOUR_VIDEO_PATH]")
    # else:
    #     cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    #     cv2.resizeWindow("test", 800,600)
    #     det = Detector()
    #     det.open(sys.argv[1])
    #     det.detect()
    cv2_flag = my_config['cv2_flag']
    if (cv2_flag == True):
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("test", 800, 600)
    det = Detector()
    det.open('./video/2.mp4')
    det.detect()