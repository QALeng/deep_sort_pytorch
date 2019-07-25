import numpy as np

from deep.feature_extractor import Extractor
from sort.nn_matching import NearestNeighborDistanceMetric
from sort.preprocessing import non_max_suppression
from sort.detection import Detection
from sort.tracker import Tracker



class DeepSort(object):
    def __init__(self, model_path,use_cuda,map_location_flag,bad_time,left_time):
        self.min_confidence = 0.3
        self.nms_max_overlap = 1.0

        self.extractor = Extractor(model_path, use_cuda=use_cuda,map_location_flag=map_location_flag)
        # 违规时间
        self.bad_time = bad_time
        max_cosine_distance = 0.2
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric,left_time=left_time)

    def update(self, bbox_xywh, confidences, ori_img, all_name,start_time):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        detections = [Detection(bbox_xywh[i], conf, features[i], all_name[i],start_time) for i, conf in enumerate(confidences) if
                      conf > self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        #          非  最大值 抑制
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)

        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections,start_time)
        # output bbox identities
        outputs = []

        return_name = []
        stay_time_all=[]
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1 :
            # #如果是不是为确认的，则跳过
            # if not track.is_confirmed():
                continue
            #只显示违规的
            if (start_time-track.start_time)<self.bad_time:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            track_id = track.track_id

            class_name = track.class_name
            print(class_name)
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int))
            return_name.append(class_name)
            part=[class_name+str(track_id),start_time-track.start_time]
            stay_time_all.append(part)
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        #暂定状态与确定状态的都可以记录其停留时间
        # stay_time_all=[ [track.class_name,track.track_id,start_time-track.start_time] \
        #                 for track in self.tracker.tracks \
        #                 if(start_time-track.start_time)>self.bad_time and track.is_confirmed]
        return outputs, return_name,stay_time_all

    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2
    
    def _get_features(self, bbox_xywh, ori_img):
        features = []
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            feature = self.extractor(im)[0]
            features.append(feature)
        if len(features):
            features = np.stack(features, axis=0)
        else:
            features = np.array([])
        return features



if __name__ == '__main__':
    pass
