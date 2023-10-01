import torch
import torch.nn as nn

def dice_score_1c(pred, target):
    smooth = 1.
    iflat = pred.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()
    dice_scores = ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
    return dice_scores


def dice_score(pred, target, num_classes):
    dice_scores = 0
    dicelist = []
    if num_classes >1:
        for i in range(num_classes):
            dice_c = dice_score_1c(pred[:,i,:,:], target[:,i,:,:])
            dice_scores += dice_c
            dicelist.append(dice_c.cpu())
    else:
        dice_scores = dice_score_1c(pred, target)
        dicelist.append(dice_scores.cpu())
    return dice_scores / num_classes, dicelist


def dice_loss(pred, target, num_classes):
    return 1 - dice_score(pred, target, num_classes)[0]


def iou_score_1c(pred, target):
    smooth = 1.
    iflat = pred.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()
    iou_scores = ((intersection + smooth) / (iflat.sum() + tflat.sum() - intersection + smooth))
    return iou_scores


def iou_score(pred, target, num_classes):
    iou_scores = 0
    ioulist = []
    if num_classes > 1:
        for i in range(num_classes):
            iou_c = iou_score_1c(pred[:,i,:,:], target[:,i,:,:])
            iou_scores += iou_c
            ioulist.append(iou_c.cpu())
    else:
        iou_scores = iou_score_1c(pred, target)
        ioulist.append(iou_scores.cpu())
    return iou_scores / num_classes, ioulist

def iou_loss(pred, target, num_classes):
    return 1 - iou_score(pred, target, num_classes)[0]

class BinaryFocalLoss(nn.Module):
    """
    参考 https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, gamma=2.0, alpha=0.25, epsilon=1.e-9):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        multi_hot_key = target
        logits = input
        # 如果模型没有做sigmoid的话，这里需要加上
        # logits = torch.sigmoid(logits)
        zero_hot_key = 1 - multi_hot_key
        loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        return loss.mean()

class FocalLoss(nn.Module):
    """
    参考 https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, gamma=2.0, alpha=1, epsilon=1.e-9, device=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha, device=device)
        else:
            self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        num_labels = input.size(-1)
        idx = target.view(-1, 1).long()
        one_hot_key = torch.zeros(idx.size(0), num_labels, dtype=torch.float32, device=idx.device)
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        one_hot_key[:, 0] = 0  # ignore 0 index.
        logits = torch.softmax(input, dim=-1)
        loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        loss = loss.sum(1)
        return loss.mean()