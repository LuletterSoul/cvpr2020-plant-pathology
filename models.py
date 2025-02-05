# @Author: yelanlan, yican
# @Date: 2020-06-16 20:42:51
# @Last Modified by:   yican
# @Last Modified time: 2020-06-16 20:42:51
# Third party libraries
from inspect import Parameter
from os import getenv
import os
import torch
import torch.nn as nn
import pretrainedmodels

from utils.logs import init_logger
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class BinaryHead(nn.Module):
    def __init__(self, num_class=4, emb_size=2048, s=16.0):
        super(BinaryHead, self).__init__()
        self.s = s
        self.fc = nn.Sequential(nn.Linear(emb_size, num_class))

    def forward(self, fea):
        fea = l2_norm(fea)
        logit = self.fc(fea) * self.s
        return logit


class se_resnext50_32x4d(nn.Module):
    def __init__(self, num_classes=7):
        super(se_resnext50_32x4d, self).__init__()
        self.model_ft = nn.Sequential(
            *list(pretrainedmodels.__dict__["se_resnext50_32x4d"](
                num_classes=1000, pretrained="imagenet").children())[:-2])
        # for param in self.model_ft.parameters():
            # param.requires_grad = False
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.model_ft.last_linear = None
        self.fea_bn = nn.BatchNorm1d(2048)
        self.fea_bn.bias.requires_grad_(False)
        self.binary_head = BinaryHead(num_classes, emb_size=2048, s=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, **kwargs):

        img_feature = self.model_ft(x)
        img_feature = self.avg_pool(img_feature)
        img_feature = img_feature.view(img_feature.size(0), -1)
        fea = self.fea_bn(img_feature)
        # fea = self.dropout(fea)
        output = self.binary_head(fea)

        return output

class ClassificationHeadArcFace(nn.Module):
    def __init__(self, num_class=4, emb_size=2048, s=1.0):
        super(ClassificationHeadArcFace, self).__init__()
        self.s = s
        # self.weight = Parameter(num_class, emb_size)
        # nn.init.xavier_uniform_(self.weight)
        self.fc = nn.Sequential(nn.Linear(emb_size, num_class, bias=False))
        # self.fc.weight = self.weight

    def forward(self, fea):
        cosine = F.linear(F.normalize(fea), F.normalize(self.fc[0].weight))
        return cosine
class se_resnext50_32x4d_arcface(nn.Module):
    def __init__(self, num_classes=7):
        super(se_resnext50_32x4d_arcface, self).__init__()
        self.model_ft = nn.Sequential(
            *list(pretrainedmodels.__dict__["se_resnext50_32x4d"](
                num_classes=1000, pretrained="imagenet").children())[:-2])
        # for param in self.model_ft.parameters():
            # param.requires_grad = False
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.model_ft.last_linear = None
        self.binary_head = ClassificationHeadArcFace(num_classes, emb_size=2048, s=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, **kwargs):

        img_feature = self.model_ft(x)
        img_feature = self.avg_pool(img_feature)
        img_feature = img_feature.view(img_feature.size(0), -1)
        # fea = self.fea_bn(img_feature)
        # fea = self.dropout(fea)
        output = self.binary_head(img_feature)

        return output


class se_resnext50_32x4d_mask(nn.Module):
    def __init__(self, num_classes=7):
        super(se_resnext50_32x4d_mask, self).__init__()
        # print(*list(pretrainedmodels.__dict__["se_resnext50_32x4d"](
                # num_classes=1000, pretrained="imagenet").children())
        se_resnext50_32x4d = list(pretrainedmodels.__dict__["se_resnext50_32x4d"](
                num_classes=1000, pretrained="imagenet").children())
        self.model_ft= [nn.Sequential(
        nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        nn.Conv2d(64, 64, kernel_size=1, bias=False),
        *se_resnext50_32x4d[0][1:])
        ] + list(se_resnext50_32x4d[1:-2]) 
        self.model_ft = nn.Sequential(*self.model_ft)
        # self.logger = init_logger("HEC", log_dir=os.getenv("EXP_DIR"))
        # self.logger.info(se_resnext50_32x4d[0])
        # self.logger.info(se_resnext50_32x4d[1])
        # self.logger.info(se_resnext50_32x4d[-1])
        # self.model_ft = nn.Sequential(
            # *list(pretrainedmodels.__dict__["se_resnext50_32x4d"](
                # num_classes=1000, pretrained="imagenet").children())[:-2])
        # for param in self.model_ft.parameters():
            # param.requires_grad = False
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.model_ft.last_linear = None
        self.fea_bn = nn.BatchNorm1d(2048)
        self.fea_bn.bias.requires_grad_(False)
        self.binary_head = BinaryHead(num_classes, emb_size=2048, s=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, roi_mask=None, ar_mask=None):
        roi_mask = roi_mask.unsqueeze(1).type_as(x)
        ar_mask = ar_mask.unsqueeze(1).type_as(x)
        # print(roi_mask.size())
        # print(ar_mask.size())
        # print(x.size())
        # print(type(x))
        # print(roi_mask.size())
        # print(type(roi_mask))
        x = torch.concat([x, roi_mask, ar_mask], dim=1)
        img_feature = self.model_ft(x)
        img_feature = self.avg_pool(img_feature)
        img_feature = img_feature.view(img_feature.size(0), -1)
        fea = self.fea_bn(img_feature)
        # fea = self.dropout(fea)
        output = self.binary_head(fea)

        return output

class se_resnext50_32x4d_mask2(nn.Module):
    def __init__(self, num_classes=7):
        super(se_resnext50_32x4d_mask2, self).__init__()
        self.model_ft = nn.Sequential(
            *list(pretrainedmodels.__dict__["se_resnext50_32x4d"](
                num_classes=1000, pretrained="imagenet").children())[:-2])
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.model_ft.last_linear = None
        self.fea_bn = nn.BatchNorm1d(2560)
        self.fea_bn.bias.requires_grad_(False)
        self.binary_head = BinaryHead(num_classes, emb_size=2560, s=1)
        self.mask_head = nn.Sequential(
            nn.Linear(1280, 512, bias=False),
            nn.Linear(512, 256, bias=False),
        )
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, roi_mask=None, ar_mask=None):
        roi_mask = roi_mask.unsqueeze(1).type_as(x).repeat(1, 3, 1, 1)
        ar_mask = ar_mask.unsqueeze(1).type_as(x).repeat(1, 3, 1, 1)

        roi_mask_feature = self.efficient_net.extract_features(roi_mask)
        ar_mask_feature = self.efficient_net.extract_features(ar_mask)

        img_feature = self.model_ft(x)
        img_feature = self.avg_pool(img_feature).view(img_feature.size(0), -1)

        roi_mask_feature = self.avg_pool(roi_mask_feature).view(roi_mask_feature.size(0), -1)
        ar_mask_feature = self.avg_pool(ar_mask_feature).view(ar_mask_feature.size(0), -1)
        
        roi_mask_feature = self.mask_head(roi_mask_feature)
        ar_mask_feature = self.mask_head(ar_mask_feature)

        # img_feature = img_feature
        features = torch.cat([img_feature, roi_mask_feature, ar_mask_feature], dim =-1)
        fea = self.fea_bn(features)
        # fea = self.dropout(fea)
        output = self.binary_head(fea)
        return output