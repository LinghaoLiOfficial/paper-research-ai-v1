import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


# class FocalLoss(nn.Module):
#
#     def __init__(self):
#
#         super(FocalLoss, self).__init__()
#
#     # 前向传播，注意我们在计算损失函数时，比如在图像分割任务中，我们需要
#     # 使用one-hot编码将多分类任务转为多个二分类任务进行计算。
#     def forward(self, preds, labels):
#
#         total_loss = 0
#         # 使用了二分类的focal loss
#         binary_focal_loss = BinaryFocalLoss()
#         logits = F.softmax(preds, dim=1)
#         # 这里shape时[B,C,W,H]，通道一就是class num
#         nums = labels.shape[1]
#         for i in range(nums):
#             loss = binary_focal_loss(logits[:, i], labels[:, i])
#             total_loss += loss
#         return total_loss / nums


# 适用于二分类的focal loss
class FocalLoss(nn.Module):

    def __init__(self, alpha=0.3, gamma=1):  # 定义alpha和gamma变量

        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    # 前向传播
    def forward(self, preds, labels):

        preds = preds.argmax(axis=1)

        eps = 1e-7  # 防止数值超出定义域

        loss_y1 = -1 * self.alpha * torch.pow((1 - preds), self.gamma) * torch.log(preds + eps) * labels

        loss_y0 = -1 * (1 - self.alpha) * torch.pow(preds, self.gamma) * torch.log(1 - preds + eps) * (1 - labels)

        loss = loss_y0 + loss_y1

        avg_loss = torch.mean(loss)

        return Variable(avg_loss, requires_grad=True)

