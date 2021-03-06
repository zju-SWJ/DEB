"""Dataset setting and data loader for USPS.
Modified from
https://github.com/mingyuliutw/CoGAN/blob/master/cogan_pytorch/src/dataset_usps.py
"""

import gzip
import os
import pickle
import urllib
from PIL import Image

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import datasets, transforms
import copy
class_num = 10

class USPS(data.Dataset):
    """USPS Dataset.
    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, train=True, transform=None, download=False):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.filename = "usps_28x28.pkl"
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None

        # download dataset.
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.data, self.targets = self.load_samples()
        if self.train:
            total_num_samples = self.targets.shape[0]
            indices = np.arange(total_num_samples)
            self.data = self.data[indices[0:self.dataset_size], ::]
            self.targets = self.targets[indices[0:self.dataset_size]]
        self.data *= 255.0
        self.data = np.squeeze(self.data).astype(np.uint8)

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.data[index], self.targets[index]
        img = Image.fromarray(img, mode='L')
        img = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        return img, label.astype("int64")

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        """Download dataset."""
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def load_samples(self):
        """Load sample images from dataset."""
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels


class USPS_idx(data.Dataset):
    """USPS Dataset.
    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, train=True, transform=None, download=False, delete=False, seed=1, fill=False):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.filename = "usps_28x28.pkl"
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None

        # download dataset.
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.data, self.targets = self.load_samples()
        if self.train:
            total_num_samples = self.targets.shape[0]
            indices = np.arange(total_num_samples)
            self.data = self.data[indices[0:self.dataset_size], ::]
            self.targets = self.targets[indices[0:self.dataset_size]]
        self.data *= 255.0
        self.data = np.squeeze(self.data).astype(np.uint8)
        count_list = []
        idx_matrix = []
        for i in range(class_num):
            count_list.append(0)
            temp = []
            idx_matrix.append(temp)
        for i in range(len(self.data)):
            count_list[self.targets[i]] += 1
        print('Original Class Distribution:', count_list)
        if delete == True:
            np.random.seed(seed)
            ran_num = np.random.rand(len(self.data))
            del_list = []
            for i in range(len(self.data)):
                if np.power((self.targets[i] + 1.0), 2) / np.power(class_num, 2) < ran_num[i]:
                    if count_list[self.targets[i]] > 1:
                        del_list.append(i)
                        count_list[self.targets[i]] -= 1
            self.data = np.delete(self.data, del_list, axis = 0)
            self.targets = np.delete(self.targets, del_list, axis = 0)
            if fill == True:
                for i in range(len(self.data)):
                    idx_matrix[self.targets[i]].append(i)
                max_num = np.max(count_list)
                for i in range(class_num):
                    gen_num = max_num - count_list[i]
                    if gen_num == 0:
                        continue
                    select_item_idx = np.random.randint(0, high = count_list[i], size = gen_num)
                    extra_data = copy.deepcopy(self.data)[idx_matrix[i]][select_item_idx]
                    extra_targets = copy.deepcopy(self.targets)[idx_matrix[i]][select_item_idx]
                    self.data = np.concatenate((self.data, extra_data), axis = 0)
                    self.targets = np.concatenate((self.targets, extra_targets), axis = 0)
            print('Imbalanced Class Distribution:', count_list)

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.data[index], self.targets[index]
        img = Image.fromarray(img, mode='L')
        img = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        return img, label.astype("int64"), index

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        """Download dataset."""
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def load_samples(self):
        """Load sample images from dataset."""
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels