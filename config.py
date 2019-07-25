# -*- coding:utf-8 -*-
import json
import sys

def readConfig(configPath,system):
    with open(configPath, 'r', encoding='utf-8') as f:
        all = f.read()
        temp = json.loads(all)[system]
        if temp['use_cuda'] == 'True':
            temp['use_cuda'] = True
        else:
            temp['use_cuda'] = False
        if temp['map_location_flag']=='True':
            temp['map_location_flag']=True
        else:
            temp['map_location_flag']=False
        if temp['cv2_flag'] == 'True':
            temp['cv2_flag'] = True
        else:
            temp['cv2_flag'] = False
    return temp


configPath='config.json'
my_config=readConfig(configPath,'centos7')
print(my_config)


