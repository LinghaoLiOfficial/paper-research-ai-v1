import torch
import torch.nn as nn
from torch.autograd import Variable


class RecallLoss(nn.Module):

    def __init__(self):

        super(RecallLoss, self).__init__()

    def forward(self, inputs, targets):

        inputs = inputs.argmax(axis=1)

        inputs = torch.sigmoid(inputs)
        true_positives = torch.sum(targets * inputs)
        possible_positives = torch.sum(targets)
        recall = true_positives / (possible_positives + 1e-7)

        return Variable(1 - recall, requires_grad=True)


