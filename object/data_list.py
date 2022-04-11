import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path
import copy
# import cv2
import torchvision

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images."))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

        self.imgs = np.array(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        target = int(target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)

class ImageList_idx(Dataset):
    def __init__(self, image_list, delete=False, fill=False, seed=1, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images."))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

        self.imgs = np.array(self.imgs)
        if delete == True:
            class_num = -1
            for i in range(len(self.imgs)):
                if int(self.imgs[i, 1]) > class_num:
                    class_num = int(self.imgs[i, 1])
            class_num += 1
            count_list = []
            idx_matrix = []
            for i in range(class_num):
                count_list.append(0)
                temp = []
                idx_matrix.append(temp)
            for i in range(len(self.imgs)):
                count_list[int(self.imgs[i, 1])] += 1
            print('Original Class Distribution:', count_list)
            np.random.seed(seed)
            ran_num = np.random.rand(len(self.imgs))
            del_list = []
            for i in range(len(self.imgs)):
                #if np.power((int(self.imgs[i, 1]) + 1.0), 2) / np.power(class_num, 2) < ran_num[i]:
                if np.power((class_num - int(self.imgs[i, 1]) + 0.0), 2) / np.power(class_num, 2) < ran_num[i]:
                    if count_list[int(self.imgs[i, 1])] > 1:
                        del_list.append(i)
                        count_list[int(self.imgs[i, 1])] -= 1
            self.imgs = np.delete(self.imgs, del_list, axis = 0)
            # self.imgs = [self.imgs[i] for i in range(len(self.imgs)) if (i not in del_list)]
            if fill == True:
                for i in range(len(self.imgs)):
                    idx_matrix[int(self.imgs[i, 1])].append(i)
                max_num = np.max(count_list)
                for i in range(class_num):
                    gen_num = max_num - count_list[i]
                    if gen_num == 0:
                        continue
                    select_item_idx = np.random.randint(0, high = count_list[i], size = gen_num)
                    extra_data = copy.deepcopy(self.imgs)[idx_matrix[i]][select_item_idx]
                    self.imgs = np.concatenate((self.imgs, extra_data), axis = 0)
            print('Imbalanced Class Distribution:', count_list)
            
        '''
        self.data = []
        self.targets = []
        for i in range(len(self.imgs)):
            path, target = self.imgs[i]
            img = self.loader(path)
            self.data.append(img)
            self.targets.append(target)
        '''


    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        target = int(target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

    def __len__(self):
        return len(self.imgs)