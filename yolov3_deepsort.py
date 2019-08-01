import cv2
from YOLO3 import YOLO3
from sort.deep_sort import DeepSort
from util import draw_bboxes


class Detector(object):
    def __init__(self,im_height,im_width,my_config):
        self.need = my_config['need']
        self.use_cuda = my_config['use_cuda']
        self.map_location_flag = my_config['map_location_flag']
        self.bad_time = my_config['bad_time']
        self.left_time = my_config['left_time']
        self.bad_time=my_config['bad_time']
        #参数相机的   长度 宽度
        #测试的时候可以 图片的长度 宽度
        self.vdo = cv2.VideoCapture()
        yolov3Flag=my_config['yolo']
        if(yolov3Flag=='yolov3'):
            self.yolo3 = YOLO3("YOLO3/cfg/yolo_v3.cfg", "YOLO3/yolov3.weights", "YOLO3/cfg/coco.names",use_cuda=self.use_cuda, is_xywh=True)
        elif(yolov3Flag=='yolov3-tiny'):
            self.yolo3 = YOLO3("YOLO3/cfg/yolov3-tiny.cfg", "YOLO3/yolov3-tiny.weights", "YOLO3/cfg/coco.names",use_cuda=self.use_cuda, is_xywh=True)
        self.track_number=my_config['track_number']
        self.deepsort_arr = [ DeepSort("deep/checkpoint/ckpt.t7",use_cuda=self.use_cuda,map_location_flag=self.map_location_flag,
                                 bad_time=self.bad_time,left_time=self.left_time) for i in range(self.track_number)]
        self.class_names = self.yolo3.class_names
        self.write_video = True
        self.record_object={}  #
        # self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_width=im_width
        # self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.im_height=im_height
        self.area = 0, 0, self.im_width, self.im_height
        self.bad_time_object=my_config['bad_time_object']


    # def detect(self,index,ori_im,start_time):
    def detect(self,class_list):
        # 参数   self.vdo.retrieve()[1]   开始时间
        xmin, ymin, xmax, ymax = self.area
        class_name = self.class_names
        need = self.need
        return_result=[]
        for class_one in class_list:
            im = class_one.ori_im[ymin:ymax, xmin:xmax, (2, 1, 0)]
            ori_im=class_one.ori_im
            # 用于检测物体
            bbox_xywh, cls_conf, cls_ids = self.yolo3(im)
            if bbox_xywh is not None:
                # mask = cls_ids == 1
                # 用于在图片上画框
                mask = [i in need for i in cls_ids]
                all_name = [str(class_one.index) + '_' + class_name[int(i)] for i in cls_ids if i in need]
                bbox_xywh = bbox_xywh[mask]
                bbox_xywh[:, 3] *= 1.2
                cls_conf = cls_conf[mask]
                outputs, total_name, stay_time = self.deepsort_arr[class_one.index].update(bbox_xywh, cls_conf, im, all_name,
                                                                                 class_one.start_time)
                #没有人超过违规时间
                if(len(stay_time)==0):
                    continue
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]  # 可以通过这里来记录时间，因为这里可以查看当前对象的id
                    ori_im = draw_bboxes(ori_im, bbox_xyxy, identities, total_name, offset=(xmin, ymin))
                part=[class_one.index,ori_im,stay_time[0],stay_time[0:self.bad_time_object]]
                return_result.append(part)

        return return_result
        # #因为有可能会出现没有图片的情况所以最好这样返回
        # if (len(stay_time) == 0):
        #     result = (ori_im, ['null', 0], ['null', 0])
        # elif (len(stay_time) == 1):
        #     result = (ori_im, stay_time[0], ['null', 0])
        # else:
        #     result = (ori_im, stay_time[0], ori_im, stay_time[1])
        # return  ori_im,stay_time
