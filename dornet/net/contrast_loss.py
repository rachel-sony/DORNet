import torch
import torch.nn as nn
from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.ablation = ablation

    def forward(self, anchor, pos, neg):

        anchor_repeated = anchor.repeat(1, 3, 1, 1)
        pos_repeated = pos.repeat(1, 3, 1, 1)
        neg_repeated = neg.repeat(1, 3, 1, 1)

        features_anchor = self.vgg(anchor_repeated)
        features_pos = self.vgg(pos_repeated)
        features_neg = self.vgg(neg_repeated)

        loss = 0

        sum_distance_anchor_pos = 0
        sum_distance_anchor_neg = 0
        for i in range(len(features_anchor)):
            # compute distance from anchor to positive
            distance_anchor_pos = self.l1(features_anchor[i], features_pos[i].detach())

            # compute distance from anchor to negative
            distance_anchor_neg = self.l1(features_anchor[i], features_neg[i].detach())

            if not self.ablation:
                contrastive = distance_anchor_pos / (distance_anchor_neg + 1e-7)
            else:
                contrastive = distance_anchor_pos

            sum_distance_anchor_pos += distance_anchor_pos
            sum_distance_anchor_neg += distance_anchor_neg

            loss += self.weights[i] * contrastive
        return {
            'loss': loss,
            'distance_anchor_pos': sum_distance_anchor_pos,
            'distance_anchor_neg': sum_distance_anchor_neg,
        }
