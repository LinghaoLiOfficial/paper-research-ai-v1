import torch
import torch.nn as nn
from torch.autograd import Variable


class PrecisionRecallTradeoffLoss(nn.Module):

    def __init__(self, alpha=0.5):

        super(PrecisionRecallTradeoffLoss, self).__init__()

        self.alpha = alpha

    def forward(self, inputs, targets):

        inputs = inputs.argmax(axis=1)

        inputs = torch.sigmoid(inputs)
        true_positives = torch.sum(targets * inputs)
        predicted_positives = torch.sum(inputs)
        possible_positives = torch.sum(targets)

        precision = true_positives / (predicted_positives + 1e-7)
        recall = true_positives / (possible_positives + 1e-7)

        precision_loss = 1 - precision
        recall_loss = 1 - recall

        total_loss = self.alpha * precision_loss + (1 - self.alpha) * recall_loss

        return Variable(total_loss, requires_grad=True)

