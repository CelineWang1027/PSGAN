#!/usr/bin/python
# -*- encoding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG as TVGG
from torchvision.models.vgg import load_state_dict_from_url, model_urls, cfgs

from ops.spectral_norm import spectral_norm as SpectralNorm
from concern.track import Track


class ResidualBlock(nn.Module):
    """Residual Block."""
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


#also called MANet
class Generator(nn.Module):
    #Encoder-Decoder architecture
    def __init__(self, conv_dim=64):
        super(Generator, self).__init__()

        encoder_layers = [nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
                          nn.InstanceNorm2d(conv_dim, affine=False), nn.ReLU(inplace=True)]

        #Down-sampling
        curr_dim = conv_dim
        for i in range(2):
            encoder_layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            encoder_layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=False))
            encoder_layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        #Bottleneck
        for i in range(3):
            encoder_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='t'))

        decoder_layers = []
        for i in range(3):
            decoder_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='t'))

        #Up-Sampling
        for i in range(2):
            decoder_layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            decoder_layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=False))
            decoder_layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        decoder_layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        decoder_layers.append(nn.Tanh())

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        self.MDNet = MDNet()
        self.AMM = AMM()

    #input-------.only source image and reference image, no mask and position information
    def forward(self, source_image, reference_image, mask_source, mask_ref, rel_pos_source, rel_pos_ref):
        source_image, reference_image, mask_source, mask_ref, rel_pos_source, rel_pos_ref = [
            x.squeeze(0) if x.ndim == 5 else x for x in
            [source_image, reference_image, mask_source, mask_ref, rel_pos_source, rel_pos_ref]]
        fm_source = self.encoder(source_image)
        fm_reference = self.MDNet(reference_image)
        morphed_fm = self.AMM(fm_source, fm_reference, mask_source, mask_ref, rel_pos_source, rel_pos_ref)
        result = self.decoder(morphed_fm)
        return result

class MDNet(nn.Module):
    '''
    Encoder-bottleneck
    '''
    def __init__(self, conv_dim=64):
        super(MDNet, self).__init__()
        layers = [nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
                  nn.InstanceNorm2d(conv_dim, affine=True), nn.ReLU(inplace=True)]

        curr_dim = conv_dim
        #Down-sampling
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        #Bottleneck
        for i in range(3):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='p'))
        self.main = nn.Sequential(*layers)

    def forward(self, reference_image):
        fm_reference = self.main(reference_image)
        return fm_reference

class AMM(nn.Module):
    def __init__(self):
        super(AMM, self).__init__()
        self.visual_feature_weight = 0.01
        self.lambda_matrix_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        self.beta_matrix_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def get_attention_map(mask_source, mask_ref, fm_source, fm_reference, rel_pos_source, rel_pos_ref):
        HW = 64 * 64
        batch_size = 3
        #get 3 part feature using mask
        channels = fm_reference.shape[1]

        mask_source_re = F.interpolate(mask_source, size=64).repeat(1, channels, 1, 1)  #(3, c, h, w)
        fm_source = fm_source.repeat(3, 1, 1, 1)
        #(3, c, h, w), 3 stands for 3 parts
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
        weight = weight.cpu()
        weight_ind = torch.LongTensor(weight.detach().numpy().nonzero())
        weight = weight.cuda()
        weight_ind = weight_ind.cuda()
        weight *= 200
        weight = F.softmax(weight, dim=-1)
        weight = weight[weight_ind[0], weight_ind[1], weight_ind[2]]

        return torch.sparse.FloatTensor(weight_ind, weight, torch.Size([3, HW, HW]))

    @staticmethod
    def atten_feature(mask_ref, attention_map, old_gamma_matrix, old_beta_matrix):
        batch_size, channels, width, height = old_gamma_matrix.size()

        mask_ref_re = F.interpolate(mask_ref, size=old_gamma_matrix.shape[2:]).repeat(1, channels, 1, 1)
        gamma_ref_re = old_gamma_matrix.repeat(3, 1, 1, 1)
        old_gamma_matrix = gamma_ref_re * mask_ref_re
        beta_ref_re = old_beta_matrix.repeat(3, 1, 1, 1)
        old_beta_matrix = beta_ref_re * mask_ref_re

        old_gamma_matrix = old_gamma_matrix.view(3, 1, -1)
        old_beta_matrix = old_beta_matrix.view(3, 1, -1)

        old_gamma_matrix = old_gamma_matrix.permute(0, 2, 1)
        old_beta_matrix = old_beta_matrix.permute(0, 2, 1)
        new_gamma_matrix = torch.bmm(attention_map.to_dense(), old_gamma_matrix)
        new_beta_matrix = torch.bmm(attention_map.to_dense(), old_beta_matrix)
        gamma = new_gamma_matrix.view(-1, 1, width, height)
        beta = new_beta_matrix.view(-1, 1, width, height)

        gamma = (gamma[0] + gamma[1] + gamma[2]).unsqueeze(0)
        beta = (beta[0] + beta[1] + beta[2]).unsqueeze(0)
        return gamma, beta

    def forward(self, fm_source, fm_reference, mask_source, mask_ref, rel_pos_source, rel_pos_ref):

        old_gamma_matrix = self.lambda_matrix_conv(fm_reference)
        old_beta_matrix = self.beta_matrix_conv(fm_reference)

        attention_map = self.get_attention_map(mask_source, mask_ref, fm_source, fm_reference, rel_pos_source, rel_pos_ref)

        gamma, beta = self.atten_feature(mask_ref, attention_map, old_gamma_matrix, old_beta_matrix)

        morphed_fm_source = fm_source * (1 + gamma) + beta

        return morphed_fm_source


    """
    def forward(self, fm_source, fm_reference):
        batch_size, channels, width, height = fm_reference.size()
        '''
        get makeup matrices->g(x) to reference image
        '''
        old_lambda_matrix = self.lambda_matrix_conv(fm_reference).view(batch_size, -1, width * height)
        old_beta_matrix = self.beta_matrix_conv(fm_reference).view(batch_size, -1, width * height)
        '''
        proj_query-------fm_source
        query the influence from all the pixels in reference image(to one pixel in source image) 
        proj_key---------fm_reference
        '''
        temp_fm_reference = fm_reference.view(batch_size, -1, height * width)
        temp_fm_source = fm_source.view(batch_size, -1, height * width).permute(0, 2, 1)

        energy = torch.bmm(temp_fm_source, temp_fm_reference)
        attention_map = self.softmax(energy)

        new_lambda_matrix = torch.bmm(old_lambda_matrix, attention_map.permute(0, 2, 1))
        new_beta_matrix = torch.bmm(old_beta_matrix, attention_map.permute(0, 2, 1))
        new_lambda_matrix = new_lambda_matrix.view(batch_size, 1, width, height)
        new_beta_matrix = new_beta_matrix.view(batch_size, 1, width, height)

        lambda_tensor = new_lambda_matrix.expand(batch_size, 256, width, height)
        beta_tensor = new_beta_matrix.expand(batch_size, 256, width, height)
        morphed_fm_source = torch.mul(lambda_tensor, fm_source)
        morphed_fm_source = torch.add(morphed_fm_source, beta_tensor)

        return morphed_fm_source
    """

class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""
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
        return out_makeup.squeeze()

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
