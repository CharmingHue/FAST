import os
import sys

if __name__ == "__main__":
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(__dir__)
    sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../')))

import torch.nn as nn
from models.utils.nas_utils import set_layer_from_config
import torch.nn.functional as F
import torch
import json


class FASTNeck(nn.Module):
    def __init__(self, reduce_layer1, reduce_layer2, reduce_layer3, reduce_layer4, asf_block, use_asf=True):
        super(FASTNeck, self).__init__()
        self.reduce_layer1 = reduce_layer1
        self.reduce_layer2 = reduce_layer2
        self.reduce_layer3 = reduce_layer3
        self.reduce_layer4 = reduce_layer4

        # self.quarter_layer = quarter_layer
        self.asf_block = asf_block
        self.use_asf = use_asf
        self._initialize_weights()
        
        # if self.use_asf is True:
        #     self.asf = self.asf_block(self.out_channels, self.out_channels // 4)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _upsample(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='nearest')   #TODO 原FAST中此处 mode='bilinear',考虑替换
    
    def forward(self, x):
        c2, c3, c4, c5 = x
        in2 = self.reduce_layer1(c2)
        in3 = self.reduce_layer2(c3)
        in4 = self.reduce_layer3(c4)
        in5 = self.reduce_layer4(c5)

        out4 = in4 + self._upsample(in5, in4)   #1/16
        out3 = in3 + self._upsample(in4, in3)   #1/8
        out2 = in2 + self._upsample(in3, in2)   #1/4

        # p5 = self.quarter_layer(in5)
        # p4 = self.quarter_layer(out4)
        # p3 = self.quarter_layer(out3)
        # p2 = self.quarter_layer(out2)
        # p5 = F.upsample(p5, scale_factor=8, mode="nearest")    #TODO 原Paddle中此处 align_mode=1,考虑替换
        # p4 = F.upsample(p4, scale_factor=4, mode="nearest")
        # p3 = F.upsample(p3, scale_factor=2, mode="nearest")
        
        p5 = F.interpolate(in5, scale_factor=8, mode="nearest")    #TODO 原Paddle中此处 align_mode=1,考虑替换
        p4 = F.interpolate(out4, scale_factor=4, mode="nearest")
        p3 = F.interpolate(out3, scale_factor=2, mode="nearest")
        p2 = out2
        fuse = torch.cat([p5, p4, p3, p2], axis=1)

        if self.use_asf is True:
            fuse = self.asf_block(fuse, [p5, p4, p3, p2])
        return fuse

    @staticmethod
    def build_from_config(config):
        reduce_layer1 = set_layer_from_config(config['reduce_layer1'])
        reduce_layer2 = set_layer_from_config(config['reduce_layer2'])
        reduce_layer3 = set_layer_from_config(config['reduce_layer3'])
        reduce_layer4 = set_layer_from_config(config['reduce_layer4'])
        
        #TODO paddle 中 nn.Conv2D 包含 weight_attr=ParamAttr(initializer=weight_attr),bias_attr=False),考虑替换
        # quarter_layer = set_layer_from_config(config['quarter_layer'])  
        asf_block = set_layer_from_config(config['asf_block'])  
        return FASTNeck(reduce_layer1, reduce_layer2, reduce_layer3, reduce_layer4, asf_block, True)
    
    
def fast_neck_asf(config, **kwargs):
    neck_config = json.load(open(config, 'r'))['neck']
    neck = FASTNeck.build_from_config(neck_config, **kwargs)
    return neck

if __name__ == "__main__":
   
    s1 = torch.randn(size=(1,64,128,128),)
    s2 = torch.randn(size=(1,128,64,64),)
    s3 = torch.randn(size=(1,256,32,32),)
    s4 = torch.randn(size=(1,512,16,16),)
    
    s = (s1, s2, s3,s4)
   
    config = "config/fast/nas-configs/fast_tiny_asf.config"
    model = fast_neck_asf(config=config)(s)
    print(model.size(), model)