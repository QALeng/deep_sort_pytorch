# #win10
# my_config={
#     #demo_yolo3_deepsort.py    cv2
#     'cv2Flag':True,
#     #YOLO3/detector.py
#     'use_cuda':False,
#     #deep/feature_extractor.py
#     'map_location_flag':True,
#     #YOLO3/yolo_utils.py
#     # 'use_cuda':False,
# }

#Centos7
my_config={
    #demo_yolo3_deepsort.py    cv2
    'cv2_flag':False,
    #YOLO3/detector.py
    'use_cuda':True,
    #deep/feature_extractor.py
    'map_location_flag':True,
    #YOLO3/yolo_utils.py
    # 'use_cuda':False,
    #main.py   摄像头的数量
    'tracker_number':1,
    #违规时间   单位s
    'bad_time':120,
    #判断是否离开时长  单位s
    'left_time':120,
    #我们需要检测的类型   对应coco.name
    #person bicycle  car
    'need':[1, 2],
}