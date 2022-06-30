import os
import yaml
import math
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.utils import data
from torchinfo import summary
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.parameter import Parameter
from ScratchModel import DepthWiseSeparableConv as Conv1dsc
from torch.nn import Conv1d, ELU, MaxPool1d, BatchNorm1d, Dropout, Linear, Flatten, Softmax, ReLU

##########################################################################################################################################
##########################################################################################################################################
#################################################### RES-TSSDNET #########################################################################
##########################################################################################################################################
##########################################################################################################################################

class RSM1D(nn.Module):
    def __init__(self, channels_in=None, channels_out=None):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.conv1 = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=channels_out, out_channels=channels_out, bias=False, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=channels_out, out_channels=channels_out, bias=False, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels_out)
        self.bn2 = nn.BatchNorm1d(channels_out)
        self.bn3 = nn.BatchNorm1d(channels_out)
        self.nin = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=1)

    def forward(self, xx):
        yy = F.relu(self.bn1(self.conv1(xx)))
        yy = F.relu(self.bn2(self.conv2(yy)))
        yy = self.conv3(yy)
        xx = self.nin(xx)
        xx = self.bn3(xx + yy)
        xx = F.relu(xx)
        return xx

class RSM2D(nn.Module):
    def __init__(self, channels_in=None, channels_out=None):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.conv1 = nn.Conv2d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=channels_out, out_channels=channels_out, bias=False, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=channels_out, out_channels=channels_out, bias=False, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels_out)
        self.bn2 = nn.BatchNorm2d(channels_out)
        self.bn3 = nn.BatchNorm2d(channels_out)
        self.nin = nn.Conv2d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=1)

    def forward(self, xx):
        yy = F.relu(self.bn1(self.conv1(xx)))
        yy = F.relu(self.bn2(self.conv2(yy)))
        yy = self.conv3(yy)
        xx = self.nin(xx)
        xx = self.bn3(xx + yy)
        xx = F.relu(xx)
        return xx

class SSDNet1D_MOD(nn.Module):  # Res-TSSDNet 1D Modified
    def __init__(self):
        super().__init__()

        self.layer_1_1 = Conv1d(in_channels = 1, out_channels = 2, kernel_size = 10, stride = 5)
        self.layer_1_2 = Conv1d(in_channels = 1, out_channels = 2, kernel_size = 65, stride = 32)
        self.layer_1_3 = Conv1d(in_channels = 1, out_channels = 2, kernel_size = 121, stride = 60)
        self.layer_1_4 = Conv1d(in_channels = 1, out_channels = 2, kernel_size = 177, stride = 58)
        self.layer_1_5 = Conv1d(in_channels = 1, out_channels = 2, kernel_size = 232, stride = 116)
        self.layer_1_6 = Conv1d(in_channels = 1, out_channels = 2, kernel_size = 288, stride = 144)
        self.layer_1_7 = Conv1d(in_channels = 1, out_channels = 2, kernel_size = 344, stride = 172)
        self.layer_1_8 = Conv1d(in_channels = 1, out_channels = 2, kernel_size = 400, stride = 200)
        self.elu_1 = ELU()
        self.batch_norm_1 = BatchNorm1d(num_features = 16)
        self.maxpool_1 = MaxPool1d(kernel_size = 3)
        
        self.RSM1 = RSM1D(channels_in=16, channels_out=32)
        self.RSM2 = RSM1D(channels_in=32, channels_out=64)
        self.RSM3 = RSM1D(channels_in=64, channels_out=128)
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        layer_1_out_1 = self.layer_1_1(x)
        layer_1_out_2 = self.layer_1_2(x)
        layer_1_out_3 = self.layer_1_3(x)
        layer_1_out_4 = self.layer_1_4(x)
        layer_1_out_5 = self.layer_1_5(x)
        layer_1_out_6 = self.layer_1_6(x)
        layer_1_out_7 = self.layer_1_7(x)
        layer_1_out_8 = self.layer_1_8(x)
        out_block_1 = [layer_1_out_1, layer_1_out_2, layer_1_out_3, layer_1_out_4, layer_1_out_5, layer_1_out_6, layer_1_out_7, layer_1_out_8]
        padded_block_1 = [out_block_1[0]]
        for i in range(1,len(out_block_1)):
            padded_block_1.append(F.pad(out_block_1[i],pad=(0,padded_block_1[0].shape[-1]-out_block_1[i].shape[-1])))
        layer_1_out = torch.cat(padded_block_1,dim=1)
        layer_1_out = self.elu_1(layer_1_out)
        layer_1_out = self.batch_norm_1(layer_1_out)
        layer_1_out = self.maxpool_1(layer_1_out)
        x = layer_1_out
        x = F.max_pool1d(x, kernel_size=4)
        x = self.RSM1(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.RSM2(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.RSM3(x)
        x = F.max_pool1d(x, kernel_size=99)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

class SSDNet1D(nn.Module):  # Res-TSSDNet 1D
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.RSM1 = RSM1D(channels_in=16, channels_out=32)
        self.RSM2 = RSM1D(channels_in=32, channels_out=64)
        self.RSM3 = RSM1D(channels_in=64, channels_out=128)
        self.RSM4 = RSM1D(channels_in=128, channels_out=128)
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        print("INITIAL {}".format(x.shape))
        x = F.relu(self.bn1(self.conv1(x)))
        print("CONV1D-BN-RELU {}".format(x.shape))
        x = F.max_pool1d(x, kernel_size=4)
        print("MAXPOOL {}".format(x.shape))
        x = self.RSM1(x)
        print("RSM1 {}".format(x.shape))
        x = F.max_pool1d(x, kernel_size=4)
        print("MAXPOOL {}".format(x.shape))
        x = self.RSM2(x)
        print("RSM2 {}".format(x.shape))
        x = F.max_pool1d(x, kernel_size=4)
        print("MAXPOOL {}".format(x.shape))
        x = self.RSM3(x)
        print("RSM3 {}".format(x.shape))
        x = F.max_pool1d(x, kernel_size=4)
        print("MAXPOOL {}".format(x.shape))
        x = self.RSM4(x)
        print("RSM4 {}".format(x.shape))
        x = F.max_pool1d(x, kernel_size=375)
        print("MAXPOOL {}".format(x.shape))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

class SSDNet2D(nn.Module):  # Res-TSSDNet 2D
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.RSM1 = RSM2D(channels_in=16, channels_out=32)
        self.RSM2 = RSM2D(channels_in=32, channels_out=64)
        self.RSM3 = RSM2D(channels_in=64, channels_out=128)
        self.RSM4 = RSM2D(channels_in=128, channels_out=128)

        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2)

        x = self.RSM1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.RSM2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.RSM3(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.RSM4(x)
        
        x = F.avg_pool2d(x, kernel_size=(27, 25))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

class DilatedCovModule(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()

        channels_out = int(channels_out/4)
        self.cv1 = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=3, dilation=1, padding=1)
        self.cv2 = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=3, dilation=2, padding=2)
        self.cv4 = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=3, dilation=4, padding=4)
        self.cv8 = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=3, dilation=8, padding=8)
        self.bn1 = nn.BatchNorm1d(channels_out)
        self.bn2 = nn.BatchNorm1d(channels_out)
        self.bn4 = nn.BatchNorm1d(channels_out)
        self.bn8 = nn.BatchNorm1d(channels_out)

    def forward(self, xx):
        xx1 = F.relu(self.bn1(self.cv1(xx)))
        xx2 = F.relu(self.bn2(self.cv2(xx)))
        xx4 = F.relu(self.bn4(self.cv4(xx)))
        xx8 = F.relu(self.bn8(self.cv8(xx)))
        yy = torch.cat((xx1, xx2, xx4, xx8), dim=1)
        return yy


class DilatedNet(nn.Module):  # Inc-TSSDNet
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16)

        self.DCM1 = DilatedCovModule(channels_in=16, channels_out=32)
        self.DCM2 = DilatedCovModule(channels_in=32, channels_out=64)
        self.DCM3 = DilatedCovModule(channels_in=64, channels_out=128)
        self.DCM4 = DilatedCovModule(channels_in=128, channels_out=128)

        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=4)

        x = F.max_pool1d(self.DCM1(x), kernel_size=4)
        x = F.max_pool1d(self.DCM2(x), kernel_size=4)
        x = F.max_pool1d(self.DCM3(x), kernel_size=4)
        x = F.max_pool1d(self.DCM4(x), kernel_size=375)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

##########################################################################################################################################
##########################################################################################################################################
#################################################### RawNet2 (Learnable Filters) #########################################################
##########################################################################################################################################
##########################################################################################################################################

class Residual_block(nn.Module):
    def __init__(self, nb_filts, first = False):
        super(Residual_block, self).__init__()
        self.first = first
        
        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features = nb_filts[0])
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        
        self.conv1 = nn.Conv1d(in_channels = nb_filts[0],
			out_channels = nb_filts[1],
			kernel_size = 3,
			padding = 1,
			stride = 1)
        
        self.bn2 = nn.BatchNorm1d(num_features = nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels = nb_filts[1],
			out_channels = nb_filts[1],
			padding = 1,
			kernel_size = 3,
			stride = 1)
        
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels = nb_filts[0],
				out_channels = nb_filts[1],
				padding = 0,
				kernel_size = 1,
				stride = 1)
            
        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)
        
    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x
            
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        
        if self.downsample:
            identity = self.conv_downsample(identity)
            
        out += identity
        out = self.mp(out)
        return out
    
class RawNet(nn.Module):
    def __init__(self, d_args, device):
        super(RawNet, self).__init__()

        
        self.device=device
        
        self.First_conv=nn.Conv1d(out_channels = d_args['filts'][0],kernel_size = d_args['first_conv'],in_channels = d_args['in_channels'],stride = 10)
        
        self.first_bn = nn.BatchNorm1d(num_features = d_args['filts'][0])
        self.selu = nn.SELU(inplace=True)
        self.block0 = nn.Sequential(Residual_block(nb_filts = d_args['filts'][1], first = True))
        self.block1 = nn.Sequential(Residual_block(nb_filts = d_args['filts'][1]))
        self.block2 = nn.Sequential(Residual_block(nb_filts = d_args['filts'][2]))
        d_args['filts'][2][0] = d_args['filts'][2][1]
        self.block3 = nn.Sequential(Residual_block(nb_filts = d_args['filts'][2]))
        self.block4 = nn.Sequential(Residual_block(nb_filts = d_args['filts'][2]))
        self.block5 = nn.Sequential(Residual_block(nb_filts = d_args['filts'][2]))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc_attention0 = self._make_attention_fc(in_features = d_args['filts'][1][-1],
            l_out_features = d_args['filts'][1][-1])
        self.fc_attention1 = self._make_attention_fc(in_features = d_args['filts'][1][-1],
            l_out_features = d_args['filts'][1][-1])
        self.fc_attention2 = self._make_attention_fc(in_features = d_args['filts'][2][-1],
            l_out_features = d_args['filts'][2][-1])
        self.fc_attention3 = self._make_attention_fc(in_features = d_args['filts'][2][-1],
            l_out_features = d_args['filts'][2][-1])
        self.fc_attention4 = self._make_attention_fc(in_features = d_args['filts'][2][-1],
            l_out_features = d_args['filts'][2][-1])
        self.fc_attention5 = self._make_attention_fc(in_features = d_args['filts'][2][-1],
            l_out_features = d_args['filts'][2][-1])

        self.bn_before_gru = nn.BatchNorm1d(num_features = d_args['filts'][2][-1])
        self.gru = nn.GRU(input_size = d_args['filts'][2][-1],
			hidden_size = d_args['gru_node'],
			num_layers = d_args['nb_gru_layer'],
			batch_first = True)

        
        self.fc1_gru = nn.Linear(in_features = d_args['gru_node'],
			out_features = d_args['nb_fc_node'])
       
        self.fc2_gru = nn.Linear(in_features = d_args['nb_fc_node'],
			out_features = d_args['nb_classes'],bias=True)
			
       
        self.sig = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x, y = None):

        x = self.First_conv(x)
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x =  self.selu(x)
        
        x0 = self.block0(x)
        y0 = self.avgpool(x0).view(x0.size(0), -1) # torch.Size([batch, filter])
        y0 = self.fc_attention0(y0)
        y0 = self.sig(y0).view(y0.size(0), y0.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x0 * y0 + y0  # (batch, filter, time) x (batch, filter, 1)
        

        x1 = self.block1(x)
        y1 = self.avgpool(x1).view(x1.size(0), -1) # torch.Size([batch, filter])
        y1 = self.fc_attention1(y1)
        y1 = self.sig(y1).view(y1.size(0), y1.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x1 * y1 + y1 # (batch, filter, time) x (batch, filter, 1)

        x2 = self.block2(x)
        y2 = self.avgpool(x2).view(x2.size(0), -1) # torch.Size([batch, filter])
        y2 = self.fc_attention2(y2)
        y2 = self.sig(y2).view(y2.size(0), y2.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x2 * y2 + y2 # (batch, filter, time) x (batch, filter, 1)

        x3 = self.block3(x)
        y3 = self.avgpool(x3).view(x3.size(0), -1) # torch.Size([batch, filter])
        y3 = self.fc_attention3(y3)
        y3 = self.sig(y3).view(y3.size(0), y3.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x3 * y3 + y3 # (batch, filter, time) x (batch, filter, 1)

        x4 = self.block4(x)
        y4 = self.avgpool(x4).view(x4.size(0), -1) # torch.Size([batch, filter])
        y4 = self.fc_attention4(y4)
        y4 = self.sig(y4).view(y4.size(0), y4.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x4 * y4 + y4 # (batch, filter, time) x (batch, filter, 1)

        x5 = self.block5(x)
        y5 = self.avgpool(x5).view(x5.size(0), -1) # torch.Size([batch, filter])
        y5 = self.fc_attention5(y5)
        y5 = self.sig(y5).view(y5.size(0), y5.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x5 * y5 + y5 # (batch, filter, time) x (batch, filter, 1)

        x = self.bn_before_gru(x)
        x = self.selu(x)
        x = x.permute(0, 2, 1)     #(batch, filt, time) >> (batch, time, filt)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:,-1,:]
        x = self.fc1_gru(x)
        x = self.fc2_gru(x)
        output=self.logsoftmax(x)
      
        return output
        
    def _make_attention_fc(self, in_features, l_out_features):

        l_fc = []
        l_fc.append(nn.Linear(in_features = in_features,out_features = l_out_features))
        return nn.Sequential(*l_fc)

    def _make_layer(self, nb_blocks, nb_filts, first = False):
        layers = []
        #def __init__(self, nb_filts, first = False):
        for i in range(nb_blocks):
            first = first if i == 0 else False
            layers.append(Residual_block(nb_filts = nb_filts,
				first = first))
            if i == 0: nb_filts[0] = nb_filts[1]
        return nn.Sequential(*layers)

################################################################################################################################################
################################################################################################################################################
############################################################ MULTI-RES CNN #####################################################################
################################################################################################################################################
################################################################################################################################################

class RawWaveFormCNN_MultiRes(nn.Module):
    def __init__(self, input_dim = 16000, num_classes = 10):
        super(RawWaveFormCNN_MultiRes, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.layer_1_1 = Conv1d(in_channels = 1, out_channels = 2, kernel_size = 10, stride = 5)
        self.layer_1_2 = Conv1d(in_channels = 1, out_channels = 2, kernel_size = 65, stride = 32)
        self.layer_1_3 = Conv1d(in_channels = 1, out_channels = 2, kernel_size = 121, stride = 60)
        self.layer_1_4 = Conv1d(in_channels = 1, out_channels = 2, kernel_size = 177, stride = 58)
        self.layer_1_5 = Conv1d(in_channels = 1, out_channels = 2, kernel_size = 232, stride = 116)
        self.layer_1_6 = Conv1d(in_channels = 1, out_channels = 2, kernel_size = 288, stride = 144)
        self.layer_1_7 = Conv1d(in_channels = 1, out_channels = 2, kernel_size = 344, stride = 172)
        self.layer_1_8 = Conv1d(in_channels = 1, out_channels = 2, kernel_size = 400, stride = 200)
        
        self.elu_1 = ELU()
        self.batch_norm_1 = BatchNorm1d(num_features = 16)
        self.maxpool_1 = MaxPool1d(kernel_size = 3)
        
        self.layer_2_1 = Conv1d(in_channels = 16, out_channels = 6, kernel_size = 3, stride = 1)
        self.layer_2_2 = Conv1d(in_channels = 16, out_channels = 6, kernel_size = 6, stride = 3)
        self.layer_2_3 = Conv1d(in_channels = 16, out_channels = 6, kernel_size = 9, stride = 4)
        self.layer_2_4 = Conv1d(in_channels = 16, out_channels = 6, kernel_size = 12, stride = 6)
        self.layer_2_5 = Conv1d(in_channels = 16, out_channels = 6, kernel_size = 15, stride = 7)
        
        self.elu_2 = ELU()
        self.batch_norm_2 = BatchNorm1d(num_features = 30)
        self.maxpool_2 = MaxPool1d(kernel_size = 3)
        
        self.layer_3_1 = Conv1d(in_channels = 30, out_channels = 16, kernel_size = 3, stride = 1)
        self.layer_3_2 = Conv1d(in_channels = 30, out_channels = 16, kernel_size = 5, stride = 2)
        self.layer_3_3 = Conv1d(in_channels = 30, out_channels = 16, kernel_size = 7, stride = 3)
        self.layer_3_4 = Conv1d(in_channels = 30, out_channels = 16, kernel_size = 9, stride = 4)
        
        self.elu_3 = ELU()
        self.batch_norm_3 = BatchNorm1d(num_features = 64)
        
        self.dense_part = nn.Sequential(OrderedDict([('flatten', Flatten()),
            ('Linear_1', Linear(in_features = 136320, out_features = 1024)),
            ('relu', ReLU(inplace = True)),
            ('dropout', Dropout(0.25)),
            ('Linear_2', Linear(in_features = 1024, out_features = self.num_classes))
        ]))

    def forward(self, x):
        layer_1_out_1 = self.layer_1_1(x)
        layer_1_out_2 = self.layer_1_2(x)
        layer_1_out_3 = self.layer_1_3(x)
        layer_1_out_4 = self.layer_1_4(x)
        layer_1_out_5 = self.layer_1_5(x)
        layer_1_out_6 = self.layer_1_6(x)
        layer_1_out_7 = self.layer_1_7(x)
        layer_1_out_8 = self.layer_1_8(x)
        
        out_block_1 = [layer_1_out_1, layer_1_out_2, layer_1_out_3, layer_1_out_4, layer_1_out_5, layer_1_out_6, layer_1_out_7, layer_1_out_8]
        padded_block_1 = [out_block_1[0]]
        for i in range(1,len(out_block_1)):
            padded_block_1.append(F.pad(out_block_1[i],pad=(0,padded_block_1[0].shape[-1]-out_block_1[i].shape[-1])))
        
        layer_1_out = torch.cat(padded_block_1,dim=1)
        layer_1_out = self.elu_1(layer_1_out)
        layer_1_out = self.batch_norm_1(layer_1_out)
        layer_1_out = self.maxpool_1(layer_1_out)
        # print(layer_1_out.shape)
        
        layer_2_out_1 = self.layer_2_1(layer_1_out)
        layer_2_out_2 = self.layer_2_2(layer_1_out)
        layer_2_out_3 = self.layer_2_3(layer_1_out)
        layer_2_out_4 = self.layer_2_4(layer_1_out)
        layer_2_out_5 = self.layer_2_5(layer_1_out)
        
        out_block_2 = [layer_2_out_1, layer_2_out_2, layer_2_out_3, layer_2_out_4, layer_2_out_5]
        padded_block_2 = [out_block_2[0]]
        for i in range(1,len(out_block_2)):
            padded_block_2.append(F.pad(out_block_2[i],pad=(0,padded_block_2[0].shape[-1]-out_block_2[i].shape[-1])))
        
        layer_2_out = torch.cat(padded_block_2,dim=1)
        layer_2_out = self.elu_2(layer_2_out)
        layer_2_out = self.batch_norm_2(layer_2_out)
        layer_2_out = self.maxpool_2(layer_2_out)
        
        layer_3_out_1 = self.layer_3_1(layer_2_out)
        layer_3_out_2 = self.layer_3_2(layer_2_out)
        layer_3_out_3 = self.layer_3_3(layer_2_out)
        layer_3_out_4 = self.layer_3_4(layer_2_out)
        # print(layer_3_out_4.shape)
        
        out_block_3 = [layer_3_out_1, layer_3_out_2, layer_3_out_3, layer_3_out_4]
        padded_block_3 = [out_block_3[0]]
        for i in range(1,len(out_block_3)):
            padded_block_3.append(F.pad(out_block_3[i],pad=(0,padded_block_3[0].shape[-1]-out_block_3[i].shape[-1])))
        
        layer_3_out = torch.cat(padded_block_3,dim=1)
        layer_3_out = self.elu_3(layer_3_out)
        layer_3_out = self.batch_norm_3(layer_3_out)
        final_out = self.dense_part(layer_3_out)
        
        if self.training:
            return final_out
        else:
            return F.softmax(final_out)
        
class RawWaveFormCNN_MultiRes_DEPTHWISE(nn.Module):
    def __init__(self, input_dim = 16000, num_classes = 10):
        super(RawWaveFormCNN_MultiRes_DEPTHWISE, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.layer_1_1 = Conv1dsc(in_channels = 1, out_channel = 2, kernel_size = 10, stride = 5)
        self.layer_1_2 = Conv1dsc(in_channels = 1, out_channel = 2, kernel_size = 65, stride = 32)
        self.layer_1_3 = Conv1dsc(in_channels = 1, out_channel = 2, kernel_size = 121, stride = 60)
        self.layer_1_4 = Conv1dsc(in_channels = 1, out_channel = 2, kernel_size = 177, stride = 58)
        self.layer_1_5 = Conv1dsc(in_channels = 1, out_channel = 2, kernel_size = 232, stride = 116)
        self.layer_1_6 = Conv1dsc(in_channels = 1, out_channel = 2, kernel_size = 288, stride = 144)
        self.layer_1_7 = Conv1dsc(in_channels = 1, out_channel = 2, kernel_size = 344, stride = 172)
        self.layer_1_8 = Conv1dsc(in_channels = 1, out_channel = 2, kernel_size = 400, stride = 200)
        
        self.elu_1 = ELU()
        self.batch_norm_1 = BatchNorm1d(num_features = 16)
        self.maxpool_1 = MaxPool1d(kernel_size = 3)
        
        self.layer_2_1 = Conv1dsc(in_channels = 16, out_channel = 6, kernel_size = 3, stride = 1)
        self.layer_2_2 = Conv1dsc(in_channels = 16, out_channel = 6, kernel_size = 6, stride = 3)
        self.layer_2_3 = Conv1dsc(in_channels = 16, out_channel = 6, kernel_size = 9, stride = 4)
        self.layer_2_4 = Conv1dsc(in_channels = 16, out_channel = 6, kernel_size = 12, stride = 6)
        self.layer_2_5 = Conv1dsc(in_channels = 16, out_channel = 6, kernel_size = 15, stride = 7)
        
        self.elu_2 = ELU()
        self.batch_norm_2 = BatchNorm1d(num_features = 30)
        self.maxpool_2 = MaxPool1d(kernel_size = 3)
        
        self.layer_3_1 = Conv1dsc(in_channels = 30, out_channel = 16, kernel_size = 3, stride = 1)
        self.layer_3_2 = Conv1dsc(in_channels = 30, out_channel = 16, kernel_size = 5, stride = 2)
        self.layer_3_3 = Conv1dsc(in_channels = 30, out_channel = 16, kernel_size = 7, stride = 3)
        self.layer_3_4 = Conv1dsc(in_channels = 30, out_channel = 16, kernel_size = 9, stride = 4)
        
        self.elu_3 = ELU()
        self.batch_norm_3 = BatchNorm1d(num_features = 64)
        
        self.dense_part = nn.Sequential(OrderedDict([('flatten', Flatten()),
            ('Linear_1', Linear(in_features = 136320, out_features = 1024)),
            ('relu', ReLU(inplace = True)),
            ('dropout', Dropout(0.25)),
            ('Linear_2', Linear(in_features = 1024, out_features = self.num_classes))
        ]))

    def forward(self, x):
        layer_1_out_1 = self.layer_1_1(x)
        layer_1_out_2 = self.layer_1_2(x)
        layer_1_out_3 = self.layer_1_3(x)
        layer_1_out_4 = self.layer_1_4(x)
        layer_1_out_5 = self.layer_1_5(x)
        layer_1_out_6 = self.layer_1_6(x)
        layer_1_out_7 = self.layer_1_7(x)
        layer_1_out_8 = self.layer_1_8(x)
        
        out_block_1 = [layer_1_out_1, layer_1_out_2, layer_1_out_3, layer_1_out_4, layer_1_out_5, layer_1_out_6, layer_1_out_7, layer_1_out_8]
        padded_block_1 = [out_block_1[0]]
        for i in range(1,len(out_block_1)):
            padded_block_1.append(F.pad(out_block_1[i],pad=(0,padded_block_1[0].shape[-1]-out_block_1[i].shape[-1])))
        
        layer_1_out = torch.cat(padded_block_1,dim=1)
        layer_1_out = self.elu_1(layer_1_out)
        layer_1_out = self.batch_norm_1(layer_1_out)
        layer_1_out = self.maxpool_1(layer_1_out)
        
        layer_2_out_1 = self.layer_2_1(layer_1_out)
        layer_2_out_2 = self.layer_2_2(layer_1_out)
        layer_2_out_3 = self.layer_2_3(layer_1_out)
        layer_2_out_4 = self.layer_2_4(layer_1_out)
        layer_2_out_5 = self.layer_2_5(layer_1_out)
        
        out_block_2 = [layer_2_out_1, layer_2_out_2, layer_2_out_3, layer_2_out_4, layer_2_out_5]
        padded_block_2 = [out_block_2[0]]
        for i in range(1,len(out_block_2)):
            padded_block_2.append(F.pad(out_block_2[i],pad=(0,padded_block_2[0].shape[-1]-out_block_2[i].shape[-1])))
        
        layer_2_out = torch.cat(padded_block_2,dim=1)
        layer_2_out = self.elu_2(layer_2_out)
        layer_2_out = self.batch_norm_2(layer_2_out)
        layer_2_out = self.maxpool_2(layer_2_out)
        
        layer_3_out_1 = self.layer_3_1(layer_2_out)
        layer_3_out_2 = self.layer_3_2(layer_2_out)
        layer_3_out_3 = self.layer_3_3(layer_2_out)
        layer_3_out_4 = self.layer_3_4(layer_2_out)
        
        out_block_3 = [layer_3_out_1, layer_3_out_2, layer_3_out_3, layer_3_out_4]
        padded_block_3 = [out_block_3[0]]
        for i in range(1,len(out_block_3)):
            padded_block_3.append(F.pad(out_block_3[i],pad=(0,padded_block_3[0].shape[-1]-out_block_3[i].shape[-1])))
        
        layer_3_out = torch.cat(padded_block_3,dim=1)
        layer_3_out = self.elu_3(layer_3_out)
        layer_3_out = self.batch_norm_3(layer_3_out)
        final_out = self.dense_part(layer_3_out)
        
        if self.training:
            return final_out
        else:
            return F.softmax(final_out)


# if __name__ == '__main__':
#     # Res_TSSDNet = SSDNet1D()
#     # Res_TSSDNet_2D = SSDNet2D()
#     # Inc_TSSDNet = DilatedNet()
    
#     # dir_yaml = os.path.splitext('model_config_RawNet')[0] + '.yaml'
#     # with open(dir_yaml, 'r') as f_yaml:
#     #         parser1 = yaml.safe_load(f_yaml)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # Raw_NetCC = RawNet(parser1['model'], device)
#     # MultiResCC = RawWaveFormCNN_Segmental(input_dim=96000,num_classes=2)
    
#     # num_params_1D = sum(i.numel() for i in Raw_NetCC.parameters() if i.requires_grad)
#     # num_params_1D = sum(i.numel() for i in Res_TSSDNet.parameters() if i.requires_grad)  # 0.35M
#     # num_params_2D = sum(i.numel() for i in Res_TSSDNet_2D.parameters() if i.requires_grad)  # 0.97M
#     # num_params_Inc = sum(i.numel() for i in Inc_TSSDNet.parameters() if i.requires_grad)  # 0.09M
#     # print('Number of learnable params: 1D_Res {}, 2D {}, 1D_Inc: {}.'.format(num_params_1D, num_params_2D, num_params_Inc))
#     print('Number of learnable params: MultiResCNN {}'.format(num_params_1D))

#     # x1 = torch.randn(2, 1, 96000)
#     # x2 = torch.randn(2, 1, 432, 400)
#     # y1 = Res_TSSDNet(x1)
#     # y2 = Res_TSSDNet_2D(x2)
#     # y3 = Inc_TSSDNet(x1)

#     # x1 = torch.randn(2, 1, 96000)
#     # y1 = Raw_NetCC(x1)
#     print('End of Program.')

# model = RawWaveFormCNN_MultiRes(input_dim=(32,1,96000),num_classes=2)
# model = SSDNet1D_MOD()
# # summary(model,(32,1,96000))
# input = torch.randn(32,1,96000)
# output = model(input)
# print(output)

# model = SSDNet1D_MOD()
# summary(model,(32,1,96000))

# model = RawWaveFormCNN_MultiRes_LOWRANK(input_dim=(32,1,96000),num_classes=2)
# summary(model,(32,1,96000))
# input = torch.randn(32,1,96000)
# output = model(input)
# print(output.shape)
# print(output)