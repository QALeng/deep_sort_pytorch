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
    'cv2Flag':False,
    #YOLO3/detector.py
    'use_cuda':True,
    #deep/feature_extractor.py
    'map_location_flag':True,
    #YOLO3/yolo_utils.py
    # 'use_cuda':False,
    #main.py
    'trackerNumber':1,
}