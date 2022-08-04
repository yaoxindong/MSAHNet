import cv2
import os
import datetime
from numpy.lib.type_check import imag
import torch
import numpy as np
import matplotlib.pyplot as plt

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class storage:
    def __init__(self, save_path, net) -> None:

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        times = datetime.datetime.now()
        storage_dir = '%04d-%02d-%02d_%02d-%02d-%02d(%s)' % (times.year, times.month, times.day, times.hour, times.minute, times.second, net)
        self.storage_dir = os.path.join(save_path, storage_dir)

        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

        self.logname = os.path.join(self.storage_dir , 'log.log') 

    def log(self, string):
        assert type(string) == str
        with open(self.logname,'a+') as f:
            f.write(string + '\n')
            print(string)


    def net(self, net, name):
        
        lists = os.listdir(self.storage_dir)
        for l in lists:
            if name[:10] in l:
                file = os.path.join(self.storage_dir, l)
                os.remove(file)


        torch.save(net, os.path.join(self.storage_dir, name))


    def image(self, case_name, image, output, label):

        i = 50

        filepath = os.path.join(self.storage_dir, case_name)
        if not os.path.exists(filepath):
            os.makedirs(filepath)


        image = image.cpu().numpy()
        label = label.cpu().numpy()
        for i in range(len(image)):
            name = 'slice_' + str(i)
            fileName = os.path.join(filepath, name)

            img = image[i]
            out = output[i]
            lab = label[i]


            draw0 = np.expand_dims(img, 2) * 255

            draw1 = cv2.cvtColor(draw0, cv2.COLOR_GRAY2BGR)
            draw2 = cv2.cvtColor(draw0, cv2.COLOR_GRAY2BGR)

            # [0, 0, 255]       # 红    主动脉
            # [64, 128, 255]    # 橙    胆囊
            # [0, 255, 255]     # 黄    左肾
            # [255, 0, 0]       # 蓝    右肾
            # [255, 128, 0]     # 青    肺
            # [255, 0, 128]     # 紫    胰腺
            # [192, 128, 255]   # 粉    脾脏
            # [0, 64, 128]      # 棕    胃
            # [0, 64, 0]        # 墨绿

            color = [[0, 0, 255], [64, 128, 255], [0, 255, 255], [255, 0, 0], [255, 128, 0], [255, 0, 128], [192, 128, 255], [0, 64, 128], [0, 64, 0]]

            for i in range(9):
                gt = np.where(out==i+1)
                for x,y in zip(gt[0], gt[1]):
                    draw1[x][y] = color[i]
                        
                gt = np.where(lab==i+1)
                for x,y in zip(gt[0], gt[1]):
                    draw2[x][y] = color[i]

            cv2.imwrite(fileName + '_Original.png', draw0)
            cv2.imwrite(fileName + '_Result.png', draw1)
            cv2.imwrite(fileName + '_Label.png', draw2)