# -*- coding: utf-8 -*-
# Wang, X., Girshick, R., Gupta, A., & He, K. (2018). 
# Non-local neural networks. CVPR
# Code from https://github.com/AlexHex7/Non-local_pytorch.git

import torch
from torch import nn
from torch.nn import functional as F


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, index=4, inter_channels=None, dimension=3, sub_sample=True, use_bn=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        assert index > 0
        self.dimension = dimension
        self.index     = index
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if use_bn:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels))
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
           
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, *inputs):
        '''
        :param x: (b, c, t, h, w)
        :return:
        ''' 
        x = inputs[self.index]
       
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        
        return list(inputs[:self.index]) + [z] + list(inputs[self.index+1:])




class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, cfg):
        index           = cfg.NONLOCAL.PARA.index
        in_channels     = cfg.NONLOCAL.PARA.in_channels   
        inter_channels  = cfg.NONLOCAL.PARA.inter_channels
        sub_sample      = cfg.NONLOCAL.PARA.sub_sample    
        use_bn          = cfg.NONLOCAL.PARA.use_bn 
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              index=index,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              use_bn=use_bn)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, cfg):
        index           = cfg.NONLOCAL.PARA.index
        in_channels     = cfg.NONLOCAL.PARA.in_channels   
        inter_channels  = cfg.NONLOCAL.PARA.inter_channels
        sub_sample      = cfg.NONLOCAL.PARA.sub_sample    
        use_bn          = cfg.NONLOCAL.PARA.use_bn 
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              index=index,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              use_bn=use_bn)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, cfg):
        index           = cfg.NONLOCAL.PARA.index
        in_channels     = cfg.NONLOCAL.PARA.in_channels   
        inter_channels  = cfg.NONLOCAL.PARA.inter_channels
        sub_sample      = cfg.NONLOCAL.PARA.sub_sample    
        use_bn          = cfg.NONLOCAL.PARA.use_bn 
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              index=index,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              use_bn=use_bn)




if __name__ == '__main__':
    from torch.autograd import Variable
    import torch

    sub_sample = True
    use_bn = True

    img = Variable(torch.zeros(2, 3, 20))
    net = NONLocalBlock1D(3, sub_sample=sub_sample, use_bn=use_bn)
    out = net(img)
    print(out.size())

    img = Variable(torch.zeros(2, 3, 20, 20))
    net = NONLocalBlock2D(3, sub_sample=sub_sample, use_bn=use_bn)
    out = net(img)
    print(out.size())

    img = Variable(torch.randn(2, 3, 10, 20, 20))
    net = NONLocalBlock3D(3, sub_sample=sub_sample, use_bn=use_bn)
    out = net(img)
    print(out.size())
