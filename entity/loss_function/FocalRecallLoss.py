import torch.nn as nn
from torch.autograd import Variable

from entity.loss_function.FocalLoss import FocalLoss


class FocalRecallLoss(nn.Module):

    def __init__(self, alpha=0.5):

        super(FocalRecallLoss, self).__init__()

        self.alpha = alpha

    def forward(self, inputs, targets):

        focal_loss_func = FocalLoss()
        focal_loss = focal_loss_func(inputs, targets)

        cross_entropy_loss_func = nn.CrossEntropyLoss()
        cross_entropy_loss = cross_entropy_loss_func(inputs, targets)

        total_loss = self.alpha * focal_loss + (1 - self.alpha) * cross_entropy_loss

        return Variable(total_loss, requires_grad=True)

