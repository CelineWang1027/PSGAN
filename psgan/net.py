#!/usr/bin/python
# -*- encoding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG as TVGG
from torchvision.models.vgg import load_state_dict_from_url, model_urls, cfgs

from ops.spectral_norm import spectral_norm as SpectralNorm
from concern.track import Track


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input

class AdaIN(nn.Module):
    def __init__(self, style_num, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_num, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalze = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalze:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalze:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalze:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_wpf=0, actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_wpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        '''
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        '''
        self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
        self.norm2 = nn.InstanceNorm2d(dim_out, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x
    '''
    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x
    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out
    '''
    def _residual(self, x):
        x = self.norm1(x)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        out = self._residual(x)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.filter = torch.tensor([[-1, -1, -1],
                                    [-1, 8., -1],
                                   [-1, -1, -1]]).to(device) / w_hpf

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=0):
        super().__init__()
        dim_in = 2**14 // img_size  #64
        self.img_size = img_size
        #self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        #self.encode = nn.ModuleList()
        #self.decode = nn.ModuleList()
        '''
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0,2),
            nn.Conv2d(dim_in, 3, 1, 1, 1, 0)
        )
        '''
        encoder_layers = [nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
                  nn.InstanceNorm2d(64, affine=False), nn.ReLU(inplace=True)]

        #down/up-sampling blocks
        #repeat_num = int(np.log2(img_size)) - 4
        repeat_num = 2
        '''
        if w_hpf > 0:
            repeat_num += 1
        '''
        #Down-sampling
        for i in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            '''
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim, w_wpf=w_hpf, upsample=True))
            '''
            encoder_layers.append(ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            #decoder_layers.append(AdainResBlk(dim_out, dim_in, style_dim, w_wpf=w_hpf, upsample=True))
            dim_in = dim_out
        '''
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True)
            )
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim, w_wpf=w_hpf, upsample=True)
            )
            dim_in = dim_out
        '''
        decoder_layers = []
        #bottleneck
        for i in range(2):
            encoder_layers.append(ResBlk(dim_out, dim_out, normalize=True))
            decoder_layers.append(AdainResBlk(dim_out, dim_out, style_dim, w_wpf=w_hpf))
        '''
        for _ in range(2):
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_wpf=w_hpf)
            )

        '''
        '''
        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
            self.hpf = HighPass(w_hpf, device)
        '''
        #dim_in = 2**14 // img_size
        #Up-sampling
        for i in range(repeat_num):
            dim_out = min(dim_in // 2, max_conv_dim)
            decoder_layers.append(AdainResBlk(dim_in, dim_out, style_dim, w_wpf=w_hpf, upsample=True))
            dim_in = dim_out

        decoder_layers.append(nn.Conv2d(dim_in, 3, kernel_size=7, stride=1, padding=3, bias=False))
        decoder_layers.append(nn.Tanh())

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        self.MDNet = MDNet()
        self.AMM = AMM()

    def forward(self, source_image, reference_image, mask_source, mask_ref, rel_pos_source, rel_pos_ref):
        source_image, reference_image, mask_source, mask_ref, rel_pos_source, rel_pos_ref = [
            x.squeeze(0) if x.ndim == 5 else x for x in
            [source_image, reference_image, mask_source, mask_ref, rel_pos_source, rel_pos_ref]]
        fm_source = self.encoder(source_image)
        fm_reference = self.MDNet(reference_image)
        morphed_fm = self.AMM(fm_source, fm_reference, mask_source, mask_ref, rel_pos_source, rel_pos_ref)
        result = self.decoder(morphed_fm)
        return result

#distill feature map of reference image
class MDNet(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super(MDNet, self).__init__()
        dim_in = 2**14 // img_size
        layers = [nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
                  nn.InstanceNorm2d(64, affine=True), nn.ReLU(inplace=True)]
        #down-sampling
        #repeat_num = int(np.log2(img_size)) - 4
        repeat_num = 2
        '''
        if w_hpf > 0:
            repeat_num += 1
        '''
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            layers.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True)
            )
            dim_in = dim_out
        #bottleneck
        for _ in range(2):
            layers.append(
                ResBlk(dim_out, dim_out, normalize=True)
            )
        '''
        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        '''
        #self.hpf = HighPass(w_hpf, device)
        self.main = nn.Sequential(*layers)

    def forward(self, reference_image):
        fm_reference = self.main(reference_image)
        return fm_reference


class AMM(nn.Module):
    def __init__(self):
        super(AMM, self).__init__()
        self.visual_feature_weight = 0.01
        self.lambda_matrix_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0,
                                            bias=False)
        self.beta_matrix_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0,
                                          bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.atten_bottleneck_g = NONLocalBlock2D()
        self.atten_bottleneck_b = NONLocalBlock2D()

    @staticmethod
    def get_attention_map(mask_source, mask_ref, fm_source, fm_reference, rel_pos_source, rel_pos_ref):
        HW = 64 * 64
        batch_size = 3
        # get 3 part feature using mask
        channels = fm_reference.shape[1]

        mask_source_re = F.interpolate(mask_source, size=64).repeat(1, channels, 1, 1)  # (3, c, h, w)
        fm_source = fm_source.repeat(3, 1, 1, 1)
        # (3, c, h, w), 3 stands for 3 parts
        fm_source = fm_source * mask_source_re

        mask_ref_re = F.interpolate(mask_ref, size=64).repeat(1, channels, 1, 1)
        fm_reference = fm_reference.repeat(3, 1, 1, 1)
        fm_reference = fm_reference * mask_ref_re

        theta_input = torch.cat((fm_source * 0.01, rel_pos_source), dim=1)
        phi_input = torch.cat((fm_reference * 0.01, rel_pos_ref), dim=1)

        theta_target = theta_input.view(batch_size, -1, HW)
        theta_target = theta_target.permute(0, 2, 1)

        phi_source = phi_input.view(batch_size, -1, HW)

        weight = torch.bmm(theta_target, phi_source)
        with torch.no_grad():
            v = weight.detach().nonzero().long().permute(1, 0)
            # This clone is required to correctly release cuda memory.
            weight_ind = v.clone()
            del v
            torch.cuda.empty_cache()
        weight *= 200
        weight = F.softmax(weight, dim=-1)
        weight = weight[weight_ind[0], weight_ind[1], weight_ind[2]]

        return torch.sparse.FloatTensor(weight_ind, weight, torch.Size([3, HW, HW]))

    @staticmethod
    def atten_feature(mask_ref, attention_map, old_gamma_matrix, old_beta_matrix, atten_module_g, atten_module_b):
        #batch_size, channels, width, height = old_gamma_matrix.size()
        channels = old_gamma_matrix.shape[1]
        mask_ref_re = F.interpolate(mask_ref, size=old_gamma_matrix.shape[2:]).repeat(1, channels, 1, 1)
        gamma_ref_re = old_gamma_matrix.repeat(3, 1, 1, 1)
        old_gamma_matrix = gamma_ref_re * mask_ref_re
        beta_ref_re = old_beta_matrix.repeat(3, 1, 1, 1)
        old_beta_matrix = beta_ref_re * mask_ref_re
        '''
        old_gamma_matrix = old_gamma_matrix.view(3, 1, -1)
        old_beta_matrix = old_beta_matrix.view(3, 1, -1)

        old_gamma_matrix = old_gamma_matrix.permute(0, 2, 1)
        old_beta_matrix = old_beta_matrix.permute(0, 2, 1)
        new_gamma_matrix = torch.bmm(attention_map.to_dense(), old_gamma_matrix)
        new_beta_matrix = torch.bmm(attention_map.to_dense(), old_beta_matrix)
        gamma = new_gamma_matrix.view(-1, 1, width, height)
        beta = new_beta_matrix.view(-1, 1, width, height)
        '''
        gamma = atten_module_g(old_gamma_matrix, attention_map)
        beta = atten_module_b(old_beta_matrix, attention_map)

        gamma = (gamma[0] + gamma[1] + gamma[2]).unsqueeze(0)
        beta = (beta[0] + beta[1] + beta[2]).unsqueeze(0)
        return gamma, beta

    def forward(self, fm_source, fm_reference, mask_source, mask_ref, rel_pos_source, rel_pos_ref):
        old_gamma_matrix = self.lambda_matrix_conv(fm_reference)
        old_beta_matrix = self.beta_matrix_conv(fm_reference)

        attention_map = self.get_attention_map(mask_source, mask_ref, fm_source, fm_reference, rel_pos_source, rel_pos_ref)

        gamma, beta = self.atten_feature(mask_ref, attention_map, old_gamma_matrix, old_beta_matrix, self.atten_bottleneck_g, self.atten_bottleneck_b)

        morphed_fm_source = fm_source * (1 + gamma) + beta

        return morphed_fm_source

class NONLocalBlock2D(nn.Module):
    def __init__(self):
        super(NONLocalBlock2D, self).__init__()
        self.g = nn.Conv2d(in_channels=1, out_channels=1,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, source, weight):
        """(b, c, h, w)
        src_diff: (3, 136, 32, 32)
        """
        batch_size = source.size(0)

        g_source = source.view(batch_size, 1, -1)  # (N, C, H*W)
        g_source = g_source.permute(0, 2, 1)  # (N, H*W, C)

        y = torch.bmm(weight.to_dense(), g_source)
        y = y.permute(0, 2, 1).contiguous()  # (N, C, H*W)
        y = y.view(batch_size, 1, *source.size()[2:])
        return y


"""
class ResidualBlock(nn.Module):
    #Residual Block.
    def __init__(self, dim_in, dim_out, net_mode=None):
        if net_mode == 'p' or (net_mode is None):
            use_affine = True
        elif net_mode == 't':
            use_affine = False
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=use_affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=use_affine)
        )

    def forward(self, x):
        return x + self.main(x)


class GetMatrix(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GetMatrix, self).__init__()
        self.get_gamma = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.get_beta = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        gamma = self.get_gamma(x)
        beta = self.get_beta(x)
        return x, gamma, beta


class NONLocalBlock2D(nn.Module):
    def __init__(self):
        super(NONLocalBlock2D, self).__init__()
        self.g = nn.Conv2d(in_channels=1, out_channels=1,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, source, weight):
        '''(b, c, h, w)
        src_diff: (3, 136, 32, 32)
        '''
        batch_size = source.size(0)

        g_source = source.view(batch_size, 1, -1)  # (N, C, H*W)
        g_source = g_source.permute(0, 2, 1)  # (N, H*W, C)

        y = torch.bmm(weight.to_dense(), g_source)
        y = y.permute(0, 2, 1).contiguous()  # (N, C, H*W)
        y = y.view(batch_size, 1, *source.size()[2:])
        return y


class Generator(nn.Module, Track):
    #Generator. Encoder-Decoder Architecture.
    def __init__(self):
        super(Generator, self).__init__()

        # -------------------------- PNet(MDNet) for obtaining makeup matrices --------------------------

        layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True)
        )
        self.pnet_in = layers

        # Down-Sampling
        curr_dim = 64
        for i in range(2):
            layers = nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True),
                nn.ReLU(inplace=True),
            )

            setattr(self, f'pnet_down_{i+1}', layers)
            curr_dim = curr_dim * 2

        # Bottleneck. All bottlenecks share the same attention module
        self.atten_bottleneck_g = NONLocalBlock2D()
        self.atten_bottleneck_b = NONLocalBlock2D()
        self.simple_spade = GetMatrix(curr_dim, 1)      # get the makeup matrix

        for i in range(3):
            setattr(self, f'pnet_bottleneck_{i+1}', ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='p'))

        # --------------------------- TNet(MANet) for applying makeup transfer ----------------------------

        self.tnet_in_conv = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.tnet_in_spade = nn.InstanceNorm2d(64, affine=False)
        self.tnet_in_relu = nn.ReLU(inplace=True)

        # Down-Sampling
        curr_dim = 64
        for i in range(2):
            setattr(self, f'tnet_down_conv_{i+1}', nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            setattr(self, f'tnet_down_spade_{i+1}', nn.InstanceNorm2d(curr_dim * 2, affine=False))
            setattr(self, f'tnet_down_relu_{i+1}', nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(6):
            setattr(self, f'tnet_bottleneck_{i+1}', ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='t'))

        # Up-Sampling
        for i in range(2):
            setattr(self, f'tnet_up_conv_{i+1}', nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            setattr(self, f'tnet_up_spade_{i+1}', nn.InstanceNorm2d(curr_dim // 2, affine=False))
            setattr(self, f'tnet_up_relu_{i+1}', nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers = nn.Sequential(
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )
        self.tnet_out = layers
        Track.__init__(self)

    @staticmethod
    def atten_feature(mask_s, weight, gamma_s, beta_s, atten_module_g, atten_module_b):
        '''
        feature size: (1, c, h, w)
        mask_c(s): (3, 1, h, w)
        diff_c: (1, 138, 256, 256)
        return: (1, c, h, w)
        '''
        channel_num = gamma_s.shape[1]

        mask_s_re = F.interpolate(mask_s, size=gamma_s.shape[2:]).repeat(1, channel_num, 1, 1)
        gamma_s_re = gamma_s.repeat(3, 1, 1, 1)
        gamma_s = gamma_s_re * mask_s_re  # (3, c, h, w)
        beta_s_re = beta_s.repeat(3, 1, 1, 1)
        beta_s = beta_s_re * mask_s_re

        gamma = atten_module_g(gamma_s, weight)  # (3, c, h, w)
        beta = atten_module_b(beta_s, weight)

        gamma = (gamma[0] + gamma[1] + gamma[2]).unsqueeze(0)  # (c, h, w) combine the three parts
        beta = (beta[0] + beta[1] + beta[2]).unsqueeze(0)
        return gamma, beta

    def get_weight(self, mask_c, mask_s, fea_c, fea_s, diff_c, diff_s):
        '''  s --> source; c --> target
        feature size: (1, 256, 64, 64)
        diff: (3, 136, 32, 32)
        '''
        HW = 64 * 64
        batch_size = 3
        assert fea_s is not None   # fea_s when i==3
        # get 3 part fea using mask
        channel_num = fea_s.shape[1]

        mask_c_re = F.interpolate(mask_c, size=64).repeat(1, channel_num, 1, 1)  # (3, c, h, w)
        fea_c = fea_c.repeat(3, 1, 1, 1)                 # (3, c, h, w)
        fea_c = fea_c * mask_c_re                        # (3, c, h, w) 3 stands for 3 parts

        mask_s_re = F.interpolate(mask_s, size=64).repeat(1, channel_num, 1, 1)
        fea_s = fea_s.repeat(3, 1, 1, 1)
        fea_s = fea_s * mask_s_re

        theta_input = torch.cat((fea_c * 0.01, diff_c), dim=1)
        phi_input = torch.cat((fea_s * 0.01, diff_s), dim=1)

        theta_target = theta_input.view(batch_size, -1, HW) # (N, C+136, H*W)
        theta_target = theta_target.permute(0, 2, 1)        # (N, H*W, C+136)

        phi_source = phi_input.view(batch_size, -1, HW)     # (N, C+136, H*W)
        self.track("before mask")

        weight = torch.bmm(theta_target, phi_source)        # (3, HW, HW)
        self.track("among bmm")
        with torch.no_grad():
            v = weight.detach().nonzero().long().permute(1, 0)
            # This clone is required to correctly release cuda memory.
            weight_ind = v.clone()
            del v
            torch.cuda.empty_cache()

        weight *= 200                                       # hyper parameters for visual feature
        weight = F.softmax(weight, dim=-1)
        weight = weight[weight_ind[0], weight_ind[1], weight_ind[2]]
        ret = torch.sparse.FloatTensor(weight_ind, weight, torch.Size([3, HW, HW]))
        self.track("after bmm")
        return ret

    def forward(self, c, s, mask_c, mask_s, diff_c, diff_s, gamma=None, beta=None, ret=False):
        c, s, mask_c, mask_s, diff_c, diff_s = [x.squeeze(0) if x.ndim == 5 else x for x in [c, s, mask_c, mask_s, diff_c, diff_s]]
        '''attention version
        c: content, stands for source image. shape: (b, c, h, w)
        s: style, stands for reference image. shape: (b, c, h, w)
        mask_list_c: lip, skin, eye. (b, 1, h, w)
        '''

        self.track("start")
        # forward c in tnet(MANet)
        c_tnet = self.tnet_in_conv(c)
        s = self.pnet_in(s)
        c_tnet = self.tnet_in_spade(c_tnet)
        c_tnet = self.tnet_in_relu(c_tnet)

        # down-sampling
        for i in range(2):
            if gamma is None:
                cur_pnet_down = getattr(self, f'pnet_down_{i+1}')
                s = cur_pnet_down(s)

            cur_tnet_down_conv = getattr(self, f'tnet_down_conv_{i+1}')
            cur_tnet_down_spade = getattr(self, f'tnet_down_spade_{i+1}')
            cur_tnet_down_relu = getattr(self, f'tnet_down_relu_{i+1}')
            c_tnet = cur_tnet_down_conv(c_tnet)
            c_tnet = cur_tnet_down_spade(c_tnet)
            c_tnet = cur_tnet_down_relu(c_tnet)
        self.track("downsampling")

        # bottleneck
        for i in range(6):
            if gamma is None and i <= 2:
                cur_pnet_bottleneck = getattr(self, f'pnet_bottleneck_{i+1}')
            cur_tnet_bottleneck = getattr(self, f'tnet_bottleneck_{i+1}')

            # get s_pnet from p and transform
            if i == 3:
                if gamma is None:               # not in test_mix
                    s, gamma, beta = self.simple_spade(s)
                    weight = self.get_weight(mask_c, mask_s, c_tnet, s, diff_c, diff_s)
                    gamma, beta = self.atten_feature(mask_s, weight, gamma, beta, self.atten_bottleneck_g, self.atten_bottleneck_b)
                    if ret:
                        return [gamma, beta]
                # else:                       # in test mode
                    # gamma, beta = param_A[0]*w + param_B[0]*(1-w), param_A[1]*w + param_B[1]*(1-w)

                c_tnet = c_tnet * (1 + gamma) + beta    # apply makeup transfer using makeup matrices

            if gamma is None and i <= 2:
                s = cur_pnet_bottleneck(s)
            c_tnet = cur_tnet_bottleneck(c_tnet)
        self.track("bottleneck")

        # up-sampling
        for i in range(2):
            cur_tnet_up_conv = getattr(self, f'tnet_up_conv_{i+1}')
            cur_tnet_up_spade = getattr(self, f'tnet_up_spade_{i+1}')
            cur_tnet_up_relu = getattr(self, f'tnet_up_relu_{i+1}')
            c_tnet = cur_tnet_up_conv(c_tnet)
            c_tnet = cur_tnet_up_spade(c_tnet)
            c_tnet = cur_tnet_up_relu(c_tnet)
        self.track("upsampling")

        c_tnet = self.tnet_out(c_tnet)
        return c_tnet
"""

class Discriminator(nn.Module):
    #Discriminator. PatchGAN.
    def __init__(self, image_size=128, conv_dim=64, repeat_num=3, norm='SN'):
        super(Discriminator, self).__init__()

        layers = []
        if norm=='SN':
            layers.append(SpectralNorm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        else:
            layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            if norm=='SN':
                layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
            else:
                layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        #k_size = int(image_size / np.power(2, repeat_num))
        if norm=='SN':
            layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=1, padding=1)))
        else:
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))
        curr_dim = curr_dim *2

        self.main = nn.Sequential(*layers)
        if norm=='SN':
            self.conv1 = SpectralNorm(nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1, bias=False))
        else:
            self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1, bias=False)

        # conv1 remain the last square size, 256*256-->30*30
        #self.conv2 = SpectralNorm(nn.Conv2d(curr_dim, 1, kernel_size=k_size, bias=False))
        #conv2 output a single number

    def forward(self, x):
        if x.ndim == 5:
            x = x.squeeze(0)
        assert x.ndim == 4, x.ndim
        h = self.main(x)
        #out_real = self.conv1(h)
        out_makeup = self.conv1(h)
        #return out_real.squeeze(), out_makeup.squeeze()
        return out_makeup

class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        
        return [out[key] for key in out_keys]


class VGG(TVGG):
    def forward(self, x):
        x = self.features(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)
