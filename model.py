import torch
import torch.nn as nn
from torchvision import models


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class resnet_model(nn.Module):

    def __init__(self, num_classes=None, include_top=False, remove_downsample=False, bnneck=False):
        super(resnet_model, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.include_top = include_top
        self.bnneck=bnneck
        ###########
        if remove_downsample:
            # remove the final downsample operation in resnet50
            self.model.layer4[0].downsample[0].stride = 1
            self.model.layer4[0].conv2.stride = 1

        if self.bnneck:
            print('with bnneck')
            self.bottleneck = nn.BatchNorm1d(2048)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)

            if self.include_top:
                self.fc = nn.Linear(2048, num_classes,bias=False)
                nn.init.kaiming_normal_(self.fc.weight, mode='fan_out')
                # nn.init.constant_(self.fc.bias, 0.)
        else:
            print('no bnneck')
            if self.include_top:
                self.fc = nn.Linear(2048, num_classes)
                nn.init.kaiming_normal_(self.fc.weight, mode='fan_out')
                nn.init.constant_(self.fc.bias, 0.)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.avgpool(x)
        feat = x.view(x.size(0), -1)
        
        if self.bnneck:
            feat = self.bottleneck(feat)

        if not self.include_top:
            return feat
        else:
            logits = self.fc(feat)
            return feat, logits