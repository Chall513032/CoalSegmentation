import os
import numpy as np
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

class Datasets(Dataset):
    def __init__(self, imgpath, maskpath, numclasses, maxiter, batchsize, is_train=True, transform=None):
        self.imgpath = imgpath
        self.maskpath = maskpath
        self.imgname = os.listdir(self.imgpath)
        self.maskname = os.listdir(self.maskpath)
        self.numclasses = numclasses
        self.datalength = maxiter*batchsize
        self.is_train = is_train
        self.transform = transform
        assert len(self.imgname) == len(self.maskname), "The number of imgs and masks should be the same"

    def __len__(self):
        if self.is_train:
            return self.datalength
        else:
            return len(self.imgname)

    def __getitem__(self, item):
        if self.is_train:
            item = random.randint(0, len(self.imgname)-1)
        assert self.imgname[item] == self.maskname[item], \
            f"The name of img{self.imgname[item]} and mask{self.maskname[item]} should be the same"
        img = Image.open(os.path.join(self.imgpath, self.imgname[item]))
        mask = Image.open(os.path.join(self.maskpath, self.maskname[item]))
        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        if self.transform != None:
            seed = torch.random.seed()
            # 该方法设置随机种子
            torch.random.manual_seed(seed)
            img = self.transform(img)
            torch.random.manual_seed(seed)
            mask = self.transform(mask)

        img = np.asarray(img)
        # img = img / 255
        if img.ndim == 2:
            img = img[np.newaxis, ...]
            img = np.concatenate([img, img, img], axis=0)
        else:
            img = img.transpose((2, 0, 1))

        mask = np.asarray(mask)
        mask_out = np.zeros([self.numclasses, *mask.shape])
        for i in range(self.numclasses):
            mask_out[i,...] = mask==i
        # mask_out = np.asarray(mask)

        return {
            "image" : torch.tensor(img).float().contiguous(),
            "mask"  : torch.tensor(mask_out).long().contiguous()
        }

