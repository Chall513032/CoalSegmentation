import os
import torch
import cv2
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms as tf
from torch.utils.data import DataLoader
import glob
from datasets import Datasets

import datetime

class segmentor(nn.Module):
    def __init__(self, backbone, decoder_head, auxiliary_head=None):
        super(segmentor, self).__init__()
        self.backbone = backbone
        self.decoder = decoder_head
        if auxiliary_head:
            self.auxiliary = auxiliary_head
        else:
            self.auxiliary = None
    def init_weights(self):
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights()

    def forward(self, x):
        features = self.backbone(x)
        logits = self.decoder(features)
        logits = F.interpolate(logits, x.shape[-2:], mode='bilinear')
        if self.auxiliary:
            auxiliary = self.auxiliary(features)
            auxiliary = F.interpolate(auxiliary, x.shape[-2:], mode='bilinear')
            return logits, auxiliary
        return logits

if __name__ == '__main__':

    imgpath = r"F:\SourceTree\library\mmsegmentation\data\marble\images\test"
    checkpointpath = r'./checkpoint/DyLoss_mylib_batch4_672/checkpoint_iter320000.pth'
    outpath = r"./data"
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    transform = tf.Compose([
        tf.ToTensor(),
        tf.ToPILImage()
    ])

    testset = Datasets(imgpath=imgpath,
                       maskpath=imgpath,
                       numclasses=3,
                       maxiter=0,
                       batchsize=1,
                       is_train=False,
                       transform=transform)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    testloader = DataLoader(dataset=testset, batch_size=1, num_workers=4, pin_memory=True)

    net = torch.load(checkpointpath)
    net.to(device)
    net.eval()
    with torch.no_grad():
        for i, input in enumerate(testloader):
            input = input['image'].to(device)
            mask = net(input)[0]
            outimg = mask.argmax(dim=1).permute(1, 2, 0).cpu()
            outimg = (outimg-outimg.min())/(outimg.max()-outimg.min())*255
            cv2.imwrite(os.path.join(outpath, f"{str(i).zfill(4)}.png"), np.asarray(outimg, 'uint8'))
            print('{}/{} [{:.{prec}}%] has done.'.format(i + 1, len(testset), (i + 1)/len(testset)*100,
                                                         prec=4))

