
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PSPHead"]

deep_features_size = {
'squeezenet':256,
'densenet'  :512,
'resnet18'  :256,
'resnet34'  :256,
'resnet50'  :1024,
'resnet101' :1024,
'resnet152' :1024
}# 倒数第三层的channel数

models_para = {
  'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512,  deep_features_size=256,  backend='squeezenet'),
  'densenet'  : lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512,  backend='densenet'),
  'resnet18'  : lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512,  deep_features_size=256,  backend='resnet18'),
  'resnet34'  : lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512,  deep_features_size=256,  backend='resnet34'),
  'resnet50'  : lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
  'resnet101' : lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
  'resnet152' : lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

class PSPHead(nn.Module):
    def __init__(self, cfg):       
        '''
        args:
            psp_size: size of input channels
            sizes : PSP module conv size ,defalut 1,2,3,6
            deep_features_size: features for classfier
        '''
        super(PSPHead, self).__init__() 
        self.psp = PSPModule(cfg.MODEL.INCHANNEL[-1], 1024, cfg.PSP.sizes)
        self.drop_1 = nn.Dropout2d(p=cfg.PSP.drop_1)

        self.up_1 = PSPUpsample(1024, 256) # scale = 2
        self.up_2 = PSPUpsample(256, 64)   # scale = 2
        self.up_3 = PSPUpsample(64, 64)    # scale = 2

        self.drop_2 = nn.Dropout2d(p=cfg.PSP.drop_2) 
        self.final = nn.Sequential(nn.Conv2d(64, cfg.MODEL.NUM_CLASSES, kernel_size=1))

        self.classifier = nn.Sequential(
            nn.Linear(cfg.MODEL.INCHANNEL[-3], 256),
            nn.ReLU(True),
            nn.Linear(256, cfg.MODEL.NUM_CLASSES))

    def forward(self, *args):
        class_f, f = args[3], args[-1] 
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)
        p = self.up_2(p)
        p = self.drop_2(p)
        p = self.up_3(p)
        p = self.drop_2(p)

        auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))
        return self.final(p), self.classifier(auxiliary)
        

class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU())

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear',align_corners=True)
        return self.conv(p)

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU(True)

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv  = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)



class PyramidPooling(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(1)
        self.pool2 = AdaptiveAvgPool2d(2)
        self.pool3 = AdaptiveAvgPool2d(3)
        self.pool4 = AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv2 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv3 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv4 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.upsample(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.upsample(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.upsample(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)
