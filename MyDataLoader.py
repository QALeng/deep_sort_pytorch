import cv2
import torch
import numpy as np

class MyDataLoader():
    def __init__(self,data,size):
        """
        自定义数据加载类
        :param data: 数据
        :param size: 图片的大小
        """
        self.length = len(data)
        self.index = -1
        self.size=size
        self.data = [self.imgFormat(one) for one in data]
        print("MyDataLoader init")

    def __getitem__(self, index):
        return self.data[index]

    def __iter__(self):
        '''
        @summary: 迭代器，生成迭代对象时调用，返回值必须是对象自己,然后for可以循环调用next方法
        '''
        return self

    def __next__(self):
        '''
        @summary: 每一次for循环都调用该方法（必须存在）
        '''
        self.index += 1
        index = self.index
        if (index >= self.length):
            raise StopIteration()
        return self.data[index]

    def __len__(self):
        return self.length

    def imgFormat(self,ori_img):
        img = ori_img.astype(np.float) / 255.
        img = cv2.resize(img, self.size)
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
        return img