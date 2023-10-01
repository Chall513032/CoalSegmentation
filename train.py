
import numpy as np
import os
import torch
import torch.nn as nn

import torch.nn.functional as F
import math
import random
import time
import argparse

from torch.utils.data import DataLoader
import torchvision.transforms as tf
from tqdm import tqdm
from warmup import WarmupQuadraticLR

from datasets import Datasets
from eval.loss import dice_score, iou_score

from mmseg.models.backbones.swin import SwinTransformer
from mmseg.models.backbones.unet import UNet, DeconvModule, InterpConv
from mmseg.models.backbones.resnet import ResNetV1c
from mmseg.models.decode_heads.sep_aspp_head import DepthwiseSeparableASPPHead
from mmseg.models.decode_heads.uper_head import UPerHead
from mmseg.models.decode_heads.fcn_head import FCNHead
from mmseg.models.losses.focal_loss import FocalLoss
from mmengine.registry import MODELS

torch.cuda.is_available()

seed = 42  # 您可以选择任何整数作为种子值
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_weights(layer):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.constant_(layer.weight, 1)
        nn.init.constant_(layer.bias, 0)


def get_model(args):
    if args.model == 'swin':
        backbone = SwinTransformer(in_channels=args.image_dim,
                                   act_cfg=dict(type='GELU'),
                                   norm_cfg=dict(type='LN', requires_grad=True))
        decoder = UPerHead(in_channels=[96, 192, 384, 768],
                           in_index=[0, 1, 2, 3],
                           dropout_ratio=0.1,
                           channels=512,
                           norm_cfg=dict(type='BN', requires_grad=True),
                           num_classes=args.label_class)
        if args.auxiliary_head:
            auxiliary = FCNHead(in_channels=384,
                                in_index=2,
                                channels=256,
                                num_convs=1,
                                norm_cfg=dict(type='BN', requires_grad=True),
                                num_classes=args.label_class, )
        else:
            auxiliary = None
    elif args.model == 'deeplab':
        backbone = ResNetV1c(depth=50,
                             num_stages=4,
                             out_indices=(0, 1, 2, 3),
                             dilations=(1, 1, 2, 4),
                             strides=(1, 2, 1, 1))
        decoder = DepthwiseSeparableASPPHead(in_channels=2048,
                                             in_index=3,
                                             channels=512,
                                             dilations=(1, 12, 24, 36),
                                             c1_in_channels=256,
                                             c1_channels=48,
                                             num_classes=args.label_class)

        if args.auxiliary_head:
            auxiliary = FCNHead(in_channels=1024,
                                in_index=2,
                                channels=256,
                                num_convs=1,
                                norm_cfg=dict(type='SyncBN', requires_grad=True),
                                num_classes=args.label_class)
        else:
            auxiliary = None
    elif args.model == 'unet':
        MODELS.register_module(name='DeconvModule', module=DeconvModule)
        MODELS.register_module(name='InterpConv', module=InterpConv)
        backbone = UNet(in_channels=3,
                        base_channels=64,
                        num_stages=5,
                        strides=(1, 1, 1, 1, 1),
                        enc_num_convs=(2, 2, 2, 2, 2),
                        dec_num_convs=(2, 2, 2, 2),
                        downsamples=(True, True, True, True),
                        enc_dilations=(1, 1, 1, 1, 1),
                        dec_dilations=(1, 1, 1, 1),
                        with_cp=False,
                        conv_cfg=None,
                        norm_cfg=dict(type='SyncBN', requires_grad=True),
                        act_cfg=dict(type='ReLU'),
                        upsample_cfg=dict(type='InterpConv'),)
        decoder = FCNHead(in_channels=64,
                          in_index=4,
                          channels=64,
                          num_convs=1,
                          concat_input=True,
                          dropout_ratio=0.1,
                          num_classes=args.label_class,
                          norm_cfg=dict(type='SyncBN', requires_grad=True))
        if args.auxiliary_head:
            auxiliary = FCNHead(in_channels=128,
                                in_index=3,
                                channels=64,
                                num_convs=1,
                                concat_input=False,
                                dropout_ratio=0.1,
                                num_classes=args.label_class,
                                norm_cfg=dict(type='SyncBN', requires_grad=True))
        else:
            auxiliary = None
    else:
        raise ValueError('Please enter a correct model')
    return backbone, decoder, auxiliary

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
        return logits, None

def get_args():
    parser = argparse.ArgumentParser(description='Train the net on images and target masks')
    parser.add_argument('--dataset_path', '-dp', type=str, required=True, help='Dataset path')
    parser.add_argument('--save_path', '-sp', type=str, required=True, help='Save path for log file and weights')
    parser.add_argument('--model', '-m', type=str, default='swin',
                        help='choice UNet/ResNet-DeeplabV3+/SwinTransformer model with unet/deeplab/swin')
    parser.add_argument('--iteration_time', '-it', type=int, default=320000, help='Number of iterations')
    parser.add_argument('--image_dim', '-id', type=int, default=3,
                        help='The channel of image, 1 for grayscale, 3 for RGB')
    parser.add_argument('--label_class', '-lc', type=int, default=4,
                        help='The class of label')
    parser.add_argument('--rotation', '-r', dest='rotation', type=int, default=10, help='Rotation angle')
    parser.add_argument('--patch_size', '-ps', dest='patch_size', type=int, default=448, help='Patch size')
    parser.add_argument('--num_workers', '-nw', dest='num_workers', type=int, default=4, help='Num workers')
    parser.add_argument('--batch_size', '-bs', dest='batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0005,
                        help='Learning rate', dest='lr')

    parser.add_argument('--auxiliary_head', '-ah', type=int, default=1, help='Auxiliary head, 1/0 to decide use or not')
    parser.add_argument('--iter_warmup', '-warm', type=int, default=1600, help='The warnup iteration, set 0 to cancel')
    parser.add_argument('--iter_report', '-ir', type=int, default=50, help='The report iteration')
    parser.add_argument('--iter_validation', '-iv', type=int, default=1600, help='The validation iteration')
    parser.add_argument('--iter_checkpoint', '-ic', type=int, default=16000, help='The checkpoint saving iteration')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    patch_size = args.patch_size

    train_transform = tf.Compose([
        tf.ToTensor(),
        tf.RandomCrop((patch_size, patch_size)),
        tf.RandomRotation(degrees=args.rotation, center=(patch_size//2, patch_size//2)),
        tf.RandomVerticalFlip(),
        tf.RandomHorizontalFlip(),
        tf.ToPILImage()
    ])
    val_transform = tf.Compose([
        tf.ToTensor(),
        tf.ToPILImage()
    ])

    trainset = Datasets(imgpath=os.path.join(args.dataset_path, 'images/training'),
                        maskpath=os.path.join(args.dataset_path, 'annotations/training'),
                        numclasses=args.label_class,
                        maxiter=args.iteration_time,
                        batchsize=args.batch_size,
                        transform=train_transform)
    valset = Datasets(imgpath=os.path.join(args.dataset_path, 'images/validation'),
                      maskpath=os.path.join(args.dataset_path, 'annotations/validation'),
                      numclasses=args.label_class,
                      maxiter=args.iteration_time,
                      batchsize=1,
                      is_train=False,
                      transform=val_transform)

    trainloader = DataLoader(dataset=trainset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             shuffle=True)
    valloader = DataLoader(dataset=valset,
                           batch_size=1,
                           num_workers=args.num_workers,
                           pin_memory=True,
                           shuffle=True)

    t = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    log_file = os.path.join(args.save_path, f'training_log{t}.txt')
    with open(log_file, "w") as f:
        f.write("Epoch, Learning Rate, Train Loss, Validation Dice, Validation IoU\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    backbone, decoder, auxiliary = get_model(args)
    net = segmentor(backbone, decoder, auxiliary)
    net.init_weights()

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
    scheduler = WarmupQuadraticLR(optimizer, args.iter_warmup, args.iteration_time)

    celoss = nn.CrossEntropyLoss()
    focal = FocalLoss()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

    net.to(device)

    iternum=0
    with tqdm(total=args.iteration_time, unit='batch', desc=f'Iter 0/{args.iteration_time}, '
                                                f'LR: xxx, '
                                                f'acc: xxx, '
                                                f'Loss_ce: xxx, '
                                                f'Loss_fl: xxx, '
                                                f'acc_aux: xxx, '
                                                f'Loss_aux: xxx') as pbar:
        net.train()
        while iternum < args.iteration_time:
            for batch in trainloader:
                # batch = next(iter(trainloader))
                if iternum % 10 == 0:
                    ceweight = (args.iteration_time - iternum) / args.iteration_time
                    flweight = 1 - ceweight

                img = batch["image"]
                mask = batch["mask"]
                img = img.to(device=device)
                mask = mask.to(device=device)
                out = net(img)

                if args.auxiliary_head:
                    pred = out[0]
                    pred_aux = out[1]
                    ce = celoss(pred, torch.argmax(mask, dim=1))
                    fl = focal(pred, mask)
                    ce_aux = celoss(pred_aux, torch.argmax(mask, dim=1))

                    losses = ceweight*ce + flweight*fl + 0.4*ce_aux
                else:
                    pred = out[0]
                    ce = celoss(pred, torch.argmax(mask, dim=1))
                    fl = focal(pred, mask)
                    losses = ceweight * ce + flweight * fl

                torch.cuda.empty_cache()
                grad_scaler.scale(losses).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.last_epoch = iternum
                scheduler.step()

                if (iternum+1) % args.iter_report==0:
                    current_lr = scheduler.get_last_lr()[0]
                    report = f'Iter {iternum + 1}/{args.iteration_time}, '\
                             f'LR: {current_lr:.10f}, ' \
                             f'acc: {math.exp(-ce.item()):.4f}, ' \
                             f'Loss_ce: {ce.item():.4f}, ' \
                             f'Loss_fl: {fl.item():.4f}'
                    if args.auxiliary_head:
                        report += f'acc_aux: {math.exp(-ce_aux.item()):.4f}, ' \
                                  f'Loss_aux: {ce_aux.item():.4f}'
                    pbar.set_description(report)
                    with open(log_file, "a") as f:
                        f.write(report+'\n')
                    pbar.update(args.iter_report)

                if (iternum+1)%args.iter_validation==0:
                    val_dice = np.array([0 for _ in range(args.label_class)], np.float32)
                    val_iou = np.array([0 for _ in range(args.label_class)], np.float32)
                    with torch.no_grad():
                        net.eval()
                        with tqdm(total=len(valset), desc="Validation:", unit='img') as pbar2:
                            for batch in valloader:
                                img = batch["image"]
                                mask = batch["mask"]
                                img = img.to(device=device, dtype=torch.float32)
                                mask = mask.to(device=device, dtype=torch.float32)

                                pred = net(img)[0]
                                pred = F.one_hot(torch.argmax(pred, dim=1), args.label_class).permute(0,3,1,2).contiguous()
                                val_dice += np.array(dice_score(pred, mask, args.label_class)[1])
                                val_iou += np.array(iou_score(pred, mask, args.label_class)[1])
                                pbar2.update(img.shape[0])
                        val_dice /= len(valset)
                        val_iou /= len(valset)
                        print(f"dice score: {np.mean(val_dice):.4f}, iou score: {np.mean(val_iou):.4f}")
                        print("\tdice\tiou")
                        for i in range(args.label_class):
                            print(f"Label{i}:{val_dice[i]:.4f}\t{val_iou[i]:.4f}")
                        with open(log_file, "a") as f:
                            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}, "
                                    f'Iter {iternum + 1}/{args.iteration_time}\n')
                            f.write(f"dice score: {np.mean(val_dice):.4f}, iou score: {np.mean(val_iou):.4f}")
                            f.write("\tdice score\tiou score\n")
                            for i in range(args.label_class):
                                f.write(f"Label{i}:{val_dice[i]:.4f}\t{val_iou[i]:.4f}\n")
                        net.train()

                if (iternum+1) % args.iter_checkpoint==0:
                    torch.save(net, os.path.join(args.save_path, 'checkpoint_iter{}.pth'.format(iternum + 1)))
                    print(f'Checkpoint {iternum + 1} saved!')
                    with open(log_file, "a") as f:
                        f.write(f'Checkpoint {iternum + 1} saved!')

                if iternum >=args.iteration_time:
                    break
                iternum += 1
    torch.save(net, os.path.join(args.save_path, 'checkpoint_iter{}.pth'.format(iternum + 1)))
    print(f'Checkpoint {iternum + 1} saved!')
