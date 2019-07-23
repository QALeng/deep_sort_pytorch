# # #win10
# my_config={
#     #demo_yolo3_deepsort.py    cv2
#     'cv2_flag':True,
#     #YOLO3/detector.py
#     'use_cuda':False,
#     #deep/feature_extractor.py
#     'map_location_flag':True,
#     #YOLO3/yolo_utils.py
#     # 'use_cuda':False,
#     #main.py   摄像头的数量
#     'tracker_number':1,
#     #违规时间   单位s
#     'bad_time':120,
#     #判断是否离开时长  单位s
#     'left_time':1,
#     #我们需要检测的类型   对应coco.name
#     #=bicycle car  person
#     'need':[1, 2,0],
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
    'bad_time':0.0000000001,
    #判断是否离开时长  单位s
    'left_time':0.0000000001,
    #我们需要检测的类型   对应coco.name
    #person bicycle
    'need':[0,1, 2],
}