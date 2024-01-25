import random
import shutil
import warnings
import os
from datetime import datetime

import matplotlib
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from functools import partial
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import cv2
import functools
from einops import rearrange

from compare.FC_EF import Unet
from compare.FC_Siam_conc import SiamUnet_conc
from compare.FC_Siam_diff import SiamUnet_diff
from compare.NestedUNet import NestedUNet
from compare.SNUNet import SNUNet_ECAM
from compare.DTCDSCN import CDNet_model
from compare.ChangeFormer import ChangeFormerV6
from compare.A2Net import A2Net
from compare.DMINet import DMINet
from compare.IFNet import DSIFN
from compare.TFI_GR import TFI_GR
from . import resnet
from compare.MobileNet import mobilenet_v2
from compare.ChangeFormer import EncoderTransformer_v3
from models.CBAM import *



class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
                         )
#尺度变换
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


#########Difference Enhancement Module###########

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        a_h = a_h.expand(-1,-1,h,w)
        a_w = a_w.expand(-1, -1, h, w)

        # out = identity * a_w * a_h

        return a_w , a_h


class CoDEM2(nn.Module):
    '''
    最新的版本
    '''
    def __init__(self,channel_dim):
        super(CoDEM2, self).__init__()

        self.channel_dim=channel_dim

        #特征连接后
        self.Conv3 = nn.Conv2d(in_channels=2*self.channel_dim,out_channels=2*self.channel_dim,kernel_size=3,stride=1,padding=1)
        #特征加和后
        # self.AvgPool = nn.functional.adaptive_avg_pool2d()
        self.Conv1 = nn.Conv2d(in_channels=2*self.channel_dim,out_channels=self.channel_dim,kernel_size=1,stride=1,padding=0)
        #最后输出
        # self.Conv1_ =nn.Conv2d(in_channels=3*self.channel_dim,out_channels=self.channel_dim,kernel_size=1,stride=1,padding=0)
        self.BN1 = nn.BatchNorm2d(2*self.channel_dim)
        self.BN2 = nn.BatchNorm2d(self.channel_dim)
        self.ReLU = nn.ReLU(inplace=True)
        #我的注意力机制
        self.coAtt_1 = CoordAtt(inp=channel_dim, oup=channel_dim, reduction=16)
        #通道,kongjian注意力机制
        # self.cam =ChannelAttention(in_channels=self.channel_dim,ratio=16)
        # self.sam = SpatialAttention()

    def forward(self,x1,x2):
        B,C,H,W = x1.shape
        f_d = torch.abs(x1-x2) #B,C,H,W
        f_c = torch.cat((x1, x2), dim=1)  # B,2C,H,W
        z_c = self.ReLU(self.BN2(self.Conv1(self.ReLU(self.BN1(self.Conv3(f_c))))))

        d_aw, d_ah = self.coAtt_1(f_d)
        z_d = f_d * d_aw * d_ah


        out = z_d + z_c

        return out



class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
#         reflection_padding = kernel_size // 2
#         self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
#         out = self.reflection_pad(x)
        out = self.conv2d(x)
        return out

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_pool_out = self.avg_pool(x)
        max_out_out = self.max_pool(x)
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool_out)))
        max_out = self.fc2(self.relu1(self.fc1(max_out_out)))
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)

class ACFF2(nn.Module):
    '''
    最新版本的ACFF 4.21,将cat改成+，去掉卷积
    '''
    def __init__(self, channel_L, channel_H):
        super(ACFF2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channel_H,out_channels=channel_L,kernel_size=1, stride=1,padding=0)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = nn.Conv2d(in_channels=2*channel_L, out_channels=channel_L, kernel_size=1, stride=1, padding=0)
        self.BN = nn.BatchNorm2d(channel_L)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(in_channels=channel_L,ratio=16)

    def forward(self, f_low,f_high):
        # _,c,h,w = f_low.shape
        #f4上采样，通道数变成原来的1/2,长宽变为原来的2倍
        f_high = self.relu(self.BN(self.conv1(self.up(f_high))))

        f_cat = f_high + f_low

        adaptive_w = self.ca(f_cat)

        out = f_low * adaptive_w+f_high*(1-adaptive_w) # B,C_l,h,w
        return out

class CatUP(nn.Module):
    def __init__(self, channel_L, channel_H):
        super(CatUP, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channel_H,out_channels=channel_L,kernel_size=1, stride=1,padding=0)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.Conv = nn.Sequential(
            nn.Conv2d(channel_L+channel_H,channel_L,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(channel_L),
            nn.ReLU(),
            nn.Conv2d(channel_L, channel_L, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(channel_L),
            )
        self.sigmod = nn.Sigmoid()
        self.ca = ChannelAttention(in_channels=channel_L,ratio=16)

    def forward(self, f_low,f_high):
        # _,c,h,w = f_low.shape
        #f4上采样，通道数变成原来的1/2,长宽变为原来的2倍
        f_high =self.up(f_high)

        f_cat = torch.cat((f_low,f_high),dim=1)
        out = self.Conv(f_cat)

        att = self.ca(out)
        output = att*out

        return output


class SupervisedAttentionModule(nn.Module):
    def __init__(self, mid_d):
        super(SupervisedAttentionModule, self).__init__()
        self.mid_d = mid_d

        self.cbam = CBAM(channel = self.mid_d)

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        context = self.cbam(x)

        x_out = self.conv2(context)

        return x_out

###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'FC_EF':
        net = Unet(input_nbr=3, label_nbr=2)

    elif args.net_G == 'FC_Siam_conc':
        net = SiamUnet_conc(input_nbr=3, label_nbr=2)

    elif args.net_G == 'FC_Siam_diff':
        net = SiamUnet_diff(input_nbr=3, label_nbr=2)

    elif args.net_G == 'UNet++':
        net = NestedUNet(num_classes=2, input_channels=6, deep_supervision=True)
    elif args.net_G == 'SNUNet':
        net = SNUNet_ECAM(in_ch=3, out_ch=2)
    elif args.net_G == 'DTCDSCN':
        net = CDNet_model(in_channels=3)
    elif args.net_G == 'ChangeFormer':
        net = ChangeFormerV6(embed_dim=args.embed_dim)
    elif args.net_G == 'A2Net':
        net = A2Net(input_nc=3, output_nc=2)
    elif args.net_G == 'DMINet':
        net = DMINet(pretrained=True)
    elif args.net_G == 'IFNet':
        net = DSIFN()
    elif args.net_G == 'TFI-GR':
        net = TFI_GR(input_nc=3, output_nc=2)

    # 新网络
    elif args.net_G == 'SEIFNet':
        net = SEIFNet(args, input_nc=3, output_nc=2) #最终版的网络


    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)


###############################################################################
# main Functions
###############################################################################


class Backbone(torch.nn.Module):
    def __init__(self, args, input_nc, output_nc,
                 resnet_stages_num=5,
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Backbone, self).__init__()


        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear',align_corners=True)
        #
        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        #
        # self.resnet_stages_num = resnet_stages_num
        #
        self.if_upsample_2x = if_upsample_2x

        #
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()



#Backbone的区别：提取特征差异
    #BIT用的
    def forward_single0(self, x):

        x = self.backbone(x)

        x0, x1, x2, x3 = x
        if self.if_upsample_2x:
            x = self.upsamplex2(x2)
        else:
            x = x2
        # output layers
        x = self.conv_pred(x)
        return x

    #SEIFNet用
    def forward_single(self, x):

        f= self.backbone(x)

        return f

    def forward_down(self,x):
        f = self.downsample(x)
        return f



class SEIFNet(Backbone):
    """
    4.4 最新版本，改进了Diff（DEM2_sobel)和ACFF2

    """

    def __init__(self, args, input_nc, output_nc,
                 decoder_softmax=False, embed_dim=64,
                 Building_Bool=False):
        super(SEIFNet, self).__init__(args, input_nc, output_nc)

        self.stage_dims = [64, 128, 256, 512]
        self.output_nc=output_nc
        self.backbone = resnet.resnet18(pretrained=True)


        self.diff1 = CoDEM2(self.stage_dims[0])
        self.diff2  = CoDEM2(self.stage_dims[1])
        self.diff3  = CoDEM2(self.stage_dims[2])
        self.diff4  = CoDEM2(self.stage_dims[3])



        #decoder
        self.ACFF3 = ACFF2(channel_L=self.stage_dims[2], channel_H=self.stage_dims[3])
        self.ACFF2 = ACFF2(channel_L=self.stage_dims[1], channel_H=self.stage_dims[2])
        self.ACFF1 = ACFF2(channel_L=self.stage_dims[0], channel_H=self.stage_dims[1])
        #(修改了一下cbam)
        self.sam_p4 = SupervisedAttentionModule(self.stage_dims[3])
        self.sam_p3 = SupervisedAttentionModule(self.stage_dims[2])
        self.sam_p2 = SupervisedAttentionModule(self.stage_dims[1])
        self.sam_p1 = SupervisedAttentionModule(self.stage_dims[0])



        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.upsample8 = nn.Upsample(scale_factor=8,mode='bilinear')
        # self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear')


        self.conv4 = nn.Conv2d(512, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(256, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1)

        self.conv_final1 = nn.Conv2d(64, output_nc, kernel_size=1)


    def forward(self, x1, x2):


        #res18
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)

        x1_0,x1_1,x1_2,x1_3 = f1
        x2_0,x2_1,x2_2,x2_3 = f2

        #diff_last
        d1 = self.diff1(x1_0, x2_0)
        d2 = self.diff2(x1_1, x2_1)
        d3 = self.diff3(x1_2, x2_2)
        d4 = self.diff4(x1_3, x2_3)

        p4 = self.sam_p4(d4)

        ACFF_43 = self.ACFF3(d3,p4)
        p3 = self.sam_p3(ACFF_43)

        ACFF_32 =self.ACFF2(d2,p3)
        p2 = self.sam_p2(ACFF_32)

        ACFF_21 = self.ACFF1(d1,p2)
        p1 = self.sam_p1(ACFF_21)

        p4_up = self.upsample8(p4)
        p4_up =self.conv4(p4_up)

        p3_up = self.upsample4(p3)
        p3_up = self.conv3(p3_up)

        p2_up = self.upsample2(p2)
        p2_up = self.conv2(p2_up)

        p= p1+p2_up+p3_up+p4_up

        p_up =self.upsample4(p)

        output = self.conv_final1(p_up)



        return output


#局部特征提取
def Local_Attention(in_channel,r):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel,out_channels=int(in_channel/r),kernel_size=1,stride=1,padding=0),
        nn.BatchNorm2d(int(in_channel/r)),
        nn.ReLU(),
        nn.Conv2d(in_channels=int(in_channel / r), out_channels=in_channel, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(in_channel)
    )

#全局特征提取
class Global_Attention(nn.Module):
    def __init__(self,in_channel=64,r=2):
        super(Global_Attention,self).__init__()
        # c,_,H,W =input.shape
        # self.r=2
        self.PWConv_1 =nn.Conv2d(in_channels=in_channel,out_channels=int(in_channel/r),kernel_size=1,stride=1,padding=0)
        self.PWConv_2 = nn.Conv2d(in_channels=int(in_channel /r), out_channels=in_channel, kernel_size=1,stride=1, padding=0)
        self.ReLU = nn.ReLU()
        self.BN_1 = nn.BatchNorm2d(int(in_channel/r))
        self.BN_2 = nn.BatchNorm2d(in_channel)

    def forward(self,input):
        x=nn.functional.adaptive_avg_pool2d(input,(1,1))
        x=self.PWConv_1(x)
        x=self.BN_1(x)
        x=self.ReLU(x)
        x=self.PWConv_2(x)
        x=self.BN_2(x)
        x=self.ReLU(x)

        #扩展到64*64的尺度
        y=x.expand_as(input)

        return y

class FDEM(nn.Module):
    """
    ST-DEM对比方法-2022 EGDE-Net
    """
    def __init__(self,channel_dim):
        super(FDEM,self).__init__()

        reduction =2
        self.channel_dim = channel_dim
        self.c_r= int(channel_dim/reduction)
        #channel reduction rate

        #local feature
        self.Local = Local_Attention(in_channel=self.channel_dim,r=reduction)
        #global feature
        self.Global = Global_Attention(in_channel= self.channel_dim,r=reduction )

        self.sigmod = nn.Sigmoid()

    def forward(self,x1,x2):

        diff=torch.abs(x1-x2)

        f_l = self.Local(diff)
        f_g = self.Global(diff)
        w = self.sigmod(f_l+f_g)
        output = w*diff + (1-w)*diff


        return output

class DifferenceModule(nn.Module):
    '''
    ST-DEM对比方法-2022 ChangeFormer
    '''
    def __init__(self,channel_dim):
        super(DifferenceModule, self).__init__()

        self.channel_dim=channel_dim

        #特征作差后
        self.conv_diff = nn.Sequential(
        nn.Conv2d(2*self.channel_dim, self.channel_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(self.channel_dim),
        nn.Conv2d(self.channel_dim, self.channel_dim, kernel_size=3, padding=1),
        nn.ReLU()
    )


    def forward(self,x1,x2):

        f_c = torch.cat((x1, x2), dim=1)  # B,2C,H,W
        out = self.conv_diff(f_c)

        return out


class DEFM(nn.Module):
    '''
    ST-DEM对比方法-2022 DEFM
    '''

    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='IN'),
                 act_cfg=dict(type='GELU')):
        super(DEFM, self).__init__()
        self.in_channels = in_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        # TODO
        conv_cfg = None
        norm_cfg = dict(type='IN')
        act_cfg = dict(type='GELU')

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels * 2, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        """Forward function."""

        f1 = self.conv1(x1)
        f2 = self.conv1(x2)
        fuse = torch.cat([f1, f2], dim=1)
        tpo = self.conv3(fuse)
        f2_ = self.warp(x2, tpo)
        output = f2_ + x1

        return output

    @staticmethod
    def warp(x, flow):
        n, c, h, w = x.size()

        norm = torch.tensor([[[[w, h]]]]).type_as(x).to(x.device)
        col = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
        row = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
        grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(x, grid, align_corners=True)
        return output

def make_prediction(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )


