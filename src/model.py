# Implementation of the U-Net model
# Paper: https://arxiv.org/pdf/1505.04597.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, batch_norm=False):
        super(UNet, self).__init__()
        self.batch_norm = batch_norm

        # Downsample
        self.conv1 = nn.Conv2d(7, 64, 3, padding=1)
        if batch_norm: self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        if batch_norm: self.bn2 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        if batch_norm: self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        if batch_norm: self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        if batch_norm: self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        if batch_norm: self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(2)

        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        if batch_norm: self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        if batch_norm: self.bn8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(2)

        # Bridge
        self.conv9 = nn.Conv2d(512, 1024, 3, padding=1)
        if batch_norm: self.bn9 = nn.BatchNorm2d(1024)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(1024, 1024, 3, padding=1)
        if batch_norm: self.bn10 = nn.BatchNorm2d(1024)
        self.relu10 = nn.ReLU(inplace=True)

        # Upsample
        self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv11 = nn.Conv2d(1024, 512, 3, padding=1)
        if batch_norm: self.bn11 = nn.BatchNorm2d(512)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        if batch_norm: self.bn12 = nn.BatchNorm2d(512)
        self.relu12 = nn.ReLU(inplace=True)

        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv13 = nn.Conv2d(512, 256, 3, padding=1)
        if batch_norm: self.bn13 = nn.BatchNorm2d(256)
        self.relu13 = nn.ReLU(inplace=True)
        self.conv14 = nn.Conv2d(256, 256, 3, padding=1)
        if batch_norm: self.bn14 = nn.BatchNorm2d(256)
        self.relu14 = nn.ReLU(inplace=True)

        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv15 = nn.Conv2d(256, 128, 3, padding=1)
        if batch_norm: self.bn15 = nn.BatchNorm2d(128)
        self.relu15 = nn.ReLU(inplace=True)
        self.conv16 = nn.Conv2d(128, 128, 3, padding=1)
        if batch_norm: self.bn16 = nn.BatchNorm2d(128)
        self.relu16 = nn.ReLU(inplace=True)

        self.upconv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv17 = nn.Conv2d(128, 64, 3, padding=1)
        if batch_norm: self.bn17 = nn.BatchNorm2d(64)
        self.relu17 = nn.ReLU(inplace=True)
        self.conv18 = nn.Conv2d(64, 64, 3, padding=1)
        if batch_norm: self.bn18 = nn.BatchNorm2d(64)
        self.relu18 = nn.ReLU(inplace=True)

        self.conv19 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Downsample
        x1 = self.conv1(x)
        if self.batch_norm: x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        if self.batch_norm: x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x2 = self.maxpool1(x1)

        x2 = self.conv3(x2)
        if self.batch_norm: x2 = self.bn3(x2)
        x2 = self.relu3(x2)
        x2 = self.conv4(x2)
        if self.batch_norm: x2 = self.bn4(x2)
        x2 = self.relu4(x2)
        x3 = self.maxpool2(x2)

        x3 = self.conv5(x3)
        if self.batch_norm: x3 = self.bn5(x3)
        x3 = self.relu5(x3)
        x3 = self.conv6(x3)
        if self.batch_norm: x3 = self.bn6(x3)
        x3 = self.relu6(x3)
        x4 = self.maxpool3(x3)

        x4 = self.conv7(x4)
        if self.batch_norm: x4 = self.bn7(x4)
        x4 = self.relu7(x4)
        x4 = self.conv8(x4)
        if self.batch_norm: x4 = self.bn8(x4)
        x4 = self.relu8(x4)
        x5 = self.maxpool4(x4)

        # Bridge
        x5 = self.conv9(x5)
        if self.batch_norm: x5 = self.bn9(x5)
        x5 = self.relu9(x5)
        x5 = self.conv10(x5)
        if self.batch_norm: x5 = self.bn10(x5)
        x5 = self.relu10(x5)

        # Upsample
        x = self.upconv1(x5)
        x = self.crop_and_concat(x, x4) # Skip connection from x4
        x = self.conv11(x)
        if self.batch_norm: x = self.bn11(x)
        x = self.relu11(x)
        x = self.conv12(x)
        if self.batch_norm: x = self.bn12(x)
        x = self.relu12(x)


        x = self.upconv2(x)
        x = self.crop_and_concat(x, x3)  # Skip connection from x3
        x = self.conv13(x)
        if self.batch_norm: x = self.bn13(x)
        x = self.relu13(x)
        x = self.conv14(x)
        if self.batch_norm: x = self.bn14(x)
        x = self.relu14(x)

        x = self.upconv3(x)
        x = self.crop_and_concat(x, x2)  # Skip connection from x2
        x = self.conv15(x)
        if self.batch_norm: x = self.bn15(x)
        x = self.relu15(x)
        x = self.conv16(x)
        if self.batch_norm: x = self.bn16(x)
        x = self.relu16(x)

        x = self.upconv4(x)
        x = self.crop_and_concat(x, x1)  # Skip connection from x1
        x = self.conv17(x)
        if self.batch_norm: x = self.bn17(x)
        x = self.relu17(x)
        x = self.conv18(x)
        if self.batch_norm: x = self.bn18(x)
        x = self.relu18(x)

        x = self.conv19(x)
        return x

    def crop_and_concat(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return torch.cat([x2, x1], dim=1)
