import torch
from torch import Tensor

def iou(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of IoU for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'IoU: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    intersection = torch.sum(input * target)
    union = torch.sum(input) + torch.sum(target) - intersection

    if union.item() == 0:
        union = intersection + epsilon

    iou_score = (intersection + epsilon) / (union + epsilon)

    if input.dim() == 2 or reduce_batch_first:
        return iou_score
    else:
        # compute and average metric for each batch element
        iou_score = 0
        for i in range(input.shape[0]):
            iou_score += iou(input[i, ...], target[i, ...])
        return iou_score / input.shape[0]


def multiclass_iou(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of IoU for all classes
    assert input.size() == target.size()
    iou_score = 0
    iou_list = []
    for channel in range(input.shape[1]):
        iou_s = iou(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
        iou_score += iou_s
        iou_list.append(iou_s.cpu())
    return iou_score / input.shape[1], iou_list


def iou_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # IoU loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_iou if multiclass else iou
    return 1 - fn(input, target, reduce_batch_first=True)[0]