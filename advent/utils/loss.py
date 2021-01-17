import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

def cross_entropy_2d(predict, target):
    """
    Args:
        predict:(n, c, h, w)
        target:(n, h, w)
    """
    assert not target.requires_grad
    assert predict.dim() == 4
    assert target.dim() == 3
    assert predict.size(0) == target.size(0), f"{predict.size(0)} vs {target.size(0)}"
    assert predict.size(2) == target.size(1), f"{predict.size(2)} vs {target.size(1)}"
    assert predict.size(3) == target.size(2), f"{predict.size(3)} vs {target.size(3)}"
    n, c, h, w = predict.size()
    target_mask = (target >= 0) * (target != 255)
    target = target[target_mask]
    if not target.data.dim():
        return Variable(torch.zeros(1))
    predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    loss = F.cross_entropy(predict, target, size_average=True)
    return loss


def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))


class WeightedBCEWithLogitsLoss(nn.Module):

    def __init__(self, size_average=True):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.size_average = size_average

    def weighted(self, input, target, weight, alpha, beta):
        if not (weight.size() == input.size()):
            raise ValueError("weight size ({}) must be the same as input size ({})".format(weight.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + (
                    (-max_val).exp() + (-input - max_val).exp()).log()  # softmax torch.Size([1, 1, 16, 32])

        if not (weight.size() == loss.size()):  # torch.Size([19, 512, 1024])
            raise ValueError("weight size ({}) must be the same as loss size ({})".format(weight.size(), loss.size()))
        if weight is not None:
            loss = alpha * loss + beta * loss * weight  # ( lamda * M + Epsilon) * Loss

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

    def forward(self, input, target, weight, alpha, beta):
        if weight is not None:
            return self.weighted(input, target, weight, alpha, beta)
        else:
            return self.weighted(input, target, None, alpha, beta)
