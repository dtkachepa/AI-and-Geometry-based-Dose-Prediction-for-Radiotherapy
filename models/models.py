import torch
import torch.nn as nn
#from mamba_ssm import Mamba
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.se3d import *

import sys
class ResidualMambaBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_sizes, stride, norm_op, nonlin, 
                 d_state, d_conv, expand, channel_token=False, embed_dim=64, heads=8, depth=1, se_type='CSE3D'):
        super(ResidualMambaBlock, self).__init__()

        # Parallel convolutional layers
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size=kernel_sizes[0], 
                               stride=stride, padding=kernel_sizes[0] // 2)
        self.conv2 = nn.Conv3d(input_channels, output_channels, kernel_size=kernel_sizes[1], 
                               stride=stride, padding=kernel_sizes[1] // 2)
        self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=kernel_sizes[2], 
                               stride=stride, padding=kernel_sizes[2] // 2)
        self.conv4 = nn.Conv3d(input_channels, output_channels, kernel_size=kernel_sizes[3], 
                               stride=stride, padding=kernel_sizes[3] // 2)

        # Normalization and Non-linearity layers
        self.norm1 = norm_op(16)
        self.norm2 = norm_op(16)
        self.norm3 = norm_op(16)
        self.norm_mamba = norm_op(output_channels)
        self.nonlin = nonlin()

        # Shortcut to handle channel mismatch
        self.shortcut = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride) \
            if input_channels != output_channels or stride != 1 else None

        # SE block selection
        if se_type == 'CSE3D':
            self.se_block = ChannelSELayer3D(output_channels, reduction_ratio=2)
        elif se_type == 'SSE3D':
            self.se_block = SpatialSELayer3D(output_channels)
        elif se_type == 'CSSE3D':
            self.se_block = ChannelSpatialSELayer3D(output_channels, reduction_ratio=2)
        elif se_type == 'PE':
            self.se_block = ProjectExciteLayer(output_channels, reduction_ratio=2)
        else:
            self.se_block = None  # No SE block


    def forward(self, x):
        # Parallel convolutional layers
        out1 = self.nonlin(self.conv1(x))
        out2 = self.nonlin(self.conv2(x))
        out3 = self.nonlin(self.conv3(x))
        out4 = self.nonlin(self.conv4(x))

        out = out1 + out2 + out3 + out4

        

        # Apply Squeeze-and-Excitation block (if specified)
        if self.se_block:
            out = self.se_block(out)

        # Add shortcut connection
        if self.shortcut:
            shortcut = self.shortcut(x)
            out += shortcut

        return out




class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=True),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = self.conv(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class Comb(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Comb, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Separate multi-scale conv branches for encoder features
        self.enc_conv_3x3 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.enc_conv_5x5 = nn.Conv3d(in_channels, in_channels, kernel_size=5, padding=2)
        self.enc_conv_7x7 = nn.Conv3d(in_channels, in_channels, kernel_size=7, padding=3)

        # Separate multi-scale conv branches for PTV features
        self.ptv_conv_3x3 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.ptv_conv_5x5 = nn.Conv3d(in_channels, in_channels, kernel_size=5, padding=2)
        self.ptv_conv_7x7 = nn.Conv3d(in_channels, in_channels, kernel_size=7, padding=3)

        # Spatial attention per scale (applied separately to enc and ptv)
        self.att_enc_3x3 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.att_enc_5x5 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.att_enc_7x7 = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        self.att_ptv_3x3 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.att_ptv_5x5 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.att_ptv_7x7 = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        # Final attention over fused features
        self.att_fusion = nn.Conv3d(in_channels, 1, kernel_size=1)

        # Expand PTV from 1 channel to in_channels
        self.channel_expand = nn.Conv3d(1, in_channels, kernel_size=1)

        self.final_out = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, enc_out, ptv):
        """
        enc_out: [B, in_channels, D, H, W]
        ptv:     [B, 1, D, H, W]
        """
        ptv_resized = F.interpolate(ptv, size=enc_out.shape[2:], mode='trilinear', align_corners=False)
        if ptv_resized.shape[1] != enc_out.shape[1]:
            ptv_resized = self.channel_expand(ptv_resized)

        # Independent multi-scale extraction for encoder and PTV
        x_3 = F.relu(self.enc_conv_3x3(enc_out))
        x_5 = F.relu(self.enc_conv_5x5(enc_out))
        x_7 = F.relu(self.enc_conv_7x7(enc_out))

        p_3 = F.relu(self.ptv_conv_3x3(ptv_resized))
        p_5 = F.relu(self.ptv_conv_5x5(ptv_resized))
        p_7 = F.relu(self.ptv_conv_7x7(ptv_resized))

        # Independent spatial attention per scale
        x_3_att = x_3 * torch.sigmoid(self.att_enc_3x3(x_3))
        x_5_att = x_5 * torch.sigmoid(self.att_enc_5x5(x_5))
        x_7_att = x_7 * torch.sigmoid(self.att_enc_7x7(x_7))

        p_3_att = p_3 * torch.sigmoid(self.att_ptv_3x3(p_3))
        p_5_att = p_5 * torch.sigmoid(self.att_ptv_5x5(p_5))
        p_7_att = p_7 * torch.sigmoid(self.att_ptv_7x7(p_7))

        # Fuse encoder and PTV at each scale, then sum
        fused = (x_3_att + p_3_att) + (x_5_att + p_5_att) + (x_7_att + p_7_att)

        # Final spatial attention over fused representation
        fused = fused * torch.sigmoid(self.att_fusion(fused))

        return self.final_out(fused)



class Encoder(nn.Module):
    def __init__(self, in_ch, list_ch, d_state, d_conv, expand, channel_token=False):
        super(Encoder, self).__init__()

        # Replace the existing SingleConv with ResidualMambaBlock
        self.encoder_1 = ResidualMambaBlock(in_ch, list_ch[1], kernel_sizes=[3, 5, 7, 9], stride=1,
                                             norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU,
                                             d_state=d_state, d_conv=d_conv, expand=expand, se_type='CSSE3D')
        
        self.encoder_2 = ResidualMambaBlock(list_ch[1], list_ch[2], kernel_sizes=[3, 5, 7, 9], stride=2,
                                             norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU,
                                             d_state=d_state, d_conv=d_conv, expand=expand, se_type='CSSE3D')
        
        self.encoder_3 = ResidualMambaBlock(list_ch[2], list_ch[3], kernel_sizes=[3, 5, 7, 9], stride=2,
                                             norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU,
                                             d_state=d_state, d_conv=d_conv, expand=expand, se_type='CSSE3D')
        
        self.encoder_4 = ResidualMambaBlock(list_ch[3], list_ch[4], kernel_sizes=[3, 5, 7, 9], stride=2,
                                             norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU,
                                             d_state=d_state, d_conv=d_conv, expand=expand, se_type='CSSE3D')
        
        self.encoder_5 = ResidualMambaBlock(list_ch[4], list_ch[5], kernel_sizes=[3, 5, 7, 9], stride=2,
                                             norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU,
                                             d_state=d_state, d_conv=d_conv, expand=expand, se_type='CSSE3D')

        self.encoder_ptv = ResidualMambaBlock(1, 1, kernel_sizes=[3, 5, 7, 9], stride = 1, 
                                              norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU,
                                              d_state=d_state, d_conv=d_conv, expand=expand, se_type='CSSE3D')
        


        self.comb2 = Comb(list_ch[1], list_ch[1])
        self.comb3 = Comb(list_ch[2], list_ch[2])
        self.comb4 = Comb(list_ch[3], list_ch[3])
        self.comb5 = Comb(list_ch[4], list_ch[4])

        

    def forward(self, x):


        PTV = x[:, 0:1, :, :, :]

        OAR = x[:, 1:8, :, :, :]

        CT = x[:, 8:9, :, :, :]


        

        out_encoder_1 = self.encoder_1(x)


        out_encoder_2 = self.encoder_2(self.comb2(out_encoder_1, PTV))
        out_encoder_3 = self.encoder_3(self.comb3(out_encoder_2, PTV))
        out_encoder_4 = self.encoder_4(self.comb4(out_encoder_3, PTV))
        out_encoder_5 = self.encoder_5(self.comb5(out_encoder_4, PTV))


        return [out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5]



class AddFeatureMaps(nn.Module):
    def __init__(self, in_channels):
        super(AddFeatureMaps, self).__init__()

        # Fusion Weights (Learnable)
        self.fusion_weight1 = nn.Parameter(torch.ones(1))  # Weight for first feature map
        self.fusion_weight2 = nn.Parameter(torch.ones(1))  # Weight for second feature map


        self.attention = ChannelSpatialSELayer3D(in_channels)

    def forward(self, F1, F2):
        """
        F1, F2: Input feature maps of shape (B, C, H, W, D)
        """
        

        f1_att = self.attention(F1)
        f2_att = self.attention(F2)

        # Weighted Fusion
        F_fused = self.fusion_weight1 * f1_att + self.fusion_weight2 * f2_att

        return F_fused

class Decoder(nn.Module):
    def __init__(self, list_ch, d_state, d_conv, expand, channel_token=False):
        super(Decoder, self).__init__()

        self.upconv_4 = UpConv(list_ch[5], list_ch[4])

        self.decoder_conv_4 = ResidualMambaBlock(list_ch[4], list_ch[4], kernel_sizes=[3, 5, 7, 9], stride=1, norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU, d_state=d_state, d_conv=d_conv, expand=expand, se_type='CSSE3D')
       
        self.upconv_3 = UpConv(list_ch[4], list_ch[3])

        self.decoder_conv_3 = ResidualMambaBlock(list_ch[3], list_ch[3], kernel_sizes=[3, 5, 7, 9], stride=1, norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU, d_state=d_state, d_conv=d_conv, expand=expand, se_type='CSSE3D')
        
        self.upconv_2 = UpConv(list_ch[3], list_ch[2])
        
        self.decoder_conv_2 = ResidualMambaBlock(list_ch[2], list_ch[2], kernel_sizes=[3, 5, 7, 9], stride=1, norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU, d_state=d_state, d_conv=d_conv, expand=expand, se_type='CSSE3D')
        
        self.upconv_1 = UpConv(list_ch[2], list_ch[1])

        self.decoder_conv_1 = ResidualMambaBlock(list_ch[1], list_ch[1], kernel_sizes=[3, 5, 7, 9], stride=1, norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU, d_state=d_state, d_conv=d_conv, expand=expand, se_type='CSSE3D')
        
        self.combiner_4 = AddFeatureMaps(in_channels=list_ch[4])
        self.combiner_3 = AddFeatureMaps(in_channels=list_ch[3])
        self.combiner_2 = AddFeatureMaps(in_channels=list_ch[2])
        self.combiner_1 = AddFeatureMaps(in_channels=list_ch[1])

      
    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5 = out_encoder
        
        out_decoder_4 = self.decoder_conv_4(self.combiner_4(self.upconv_4(out_encoder_5), out_encoder_4))

        out_decoder_3 = self.decoder_conv_3(self.combiner_3(self.upconv_3(out_decoder_4), out_encoder_3))
        
        out_decoder_2 = self.decoder_conv_2(self.combiner_2(self.upconv_2(out_decoder_3), out_encoder_2))
        
        out_decoder_1 = self.decoder_conv_1(self.combiner_1(self.upconv_1(out_decoder_2), out_encoder_1))
        

        return out_decoder_1

class BaseUNet(nn.Module):
    def __init__(self, in_ch, list_ch, d_state, d_conv, expand, channel_token=False):
        super(BaseUNet, self).__init__()
        self.encoder = Encoder(in_ch, list_ch, d_state, d_conv, expand, channel_token)
        self.decoder = Decoder(list_ch, d_state, d_conv, expand, channel_token)

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)
        return out_decoder

class Model_MTASP(nn.Module):
    def __init__(self, in_ch, out_ch, list_ch_A, list_ch_B, d_state, d_conv, expand, channel_token=False):
        super(Model_MTASP, self).__init__()
        self.net_A = BaseUNet(in_ch, list_ch_A, d_state, d_conv, expand, channel_token)
        self.conv_out_A = nn.Conv3d(list_ch_A[1], out_ch, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out_net_A = self.net_A(x)
        
        output_A = self.conv_out_A(out_net_A)
        return output_A

if __name__ == '__main__':
    model = Model_MTASP(in_ch=9, out_ch=1,
                  list_ch_A=[-1, 16, 32, 64, 128, 256],
                  list_ch_B=[-1, 32, 64, 128, 256, 512],
                  d_state=16, d_conv=4, expand=2, channel_token=False)

    sample_input = torch.randn(2, 9, 128, 128, 128)


    model(sample_input)
    #a = np.sum(np.prod(v.size()) for v in model.parameters())*4e-6
    #print(a)

