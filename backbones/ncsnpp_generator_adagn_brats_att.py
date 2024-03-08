# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ---------------------------------------------------------------

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
''' Codes adapted from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ncsnpp.py
'''

from . import utils, layers, layerspp, dense_layer
import torch.nn as nn
import functools
import torch
import numpy as np
import torch.nn.functional as F

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp_Adagn
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp_Adagn
ResnetBlockBigGAN_one = layerspp.ResnetBlockBigGANpp_Adagn_one
ResnetBlockBigGAN_cond = layerspp.ResnetBlockBigGANpp_Adagn_autoen
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
default_initializer = layers.default_init
dense = dense_layer.dense


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class SE_Attention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False),
                                nn.SiLU(),
                                nn.Linear(channel // reduction, channel, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


@utils.register_model(name='ncsnpp')
class NCSNpp(nn.Module):
    """NCSN++ model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.not_use_tanh = config.not_use_tanh
        self.act = act = nn.SiLU()
        self.z_emb_dim = z_emb_dim = config.z_emb_dim

        self.nf = nf = config.num_channels_dae
        ch_mult = config.ch_mult
        self.num_res_blocks = num_res_blocks = config.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.attn_resolutions
        dropout = config.dropout
        resamp_with_conv = config.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [config.image_size // (2 ** i) for i in range(num_resolutions)]

        self.conditional = conditional = config.conditional  # noise-conditional
        fir = config.fir
        fir_kernel = config.fir_kernel
        self.skip_rescale = skip_rescale = config.skip_rescale
        self.resblock_type = resblock_type = config.resblock_type.lower()
        self.progressive = progressive = config.progressive.lower()
        self.progressive_input = progressive_input = config.progressive_input.lower()
        self.embedding_type = embedding_type = config.embedding_type.lower()
        init_scale = 0.
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        assert embedding_type in ['fourier', 'positional']
        combine_method = config.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        modules = []
        cond_module = []
        # timestep/noise_level embedding; only for continuous training
        if embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            # assert config.training.continuous, "Fourier features are only used for continuous training."

            modules.append(layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=config.fourier_scale
            ))
            cond_module.append(layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=config.fourier_scale
            ))
            embed_dim = 2 * nf

        elif embedding_type == 'positional':
            embed_dim = nf

        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

            cond_module.append(nn.Linear(embed_dim, nf * 4))
            cond_module[-1].weight.data = default_initializer()(cond_module[-1].weight.shape)
            nn.init.zeros_(cond_module[-1].bias)
            cond_module.append(nn.Linear(nf * 4, nf * 4))
            cond_module[-1].weight.data = default_initializer()(cond_module[-1].weight.shape)
            nn.init.zeros_(cond_module[-1].bias)

        AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale)

        Upsample = functools.partial(layerspp.Upsample,
                                     with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == 'output_skip':
            self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(layerspp.Upsample,
                                                 fir=fir, fir_kernel=fir_kernel, with_conv=True)

        Downsample = functools.partial(layerspp.Downsample,
                                       with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            pyramid_downsample = functools.partial(layerspp.Downsample,
                                                   fir=fir, fir_kernel=fir_kernel, with_conv=True)

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM,
                                            act=act,
                                            dropout=dropout,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4,
                                            zemb_dim=z_emb_dim)

        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                            act=act,
                                            dropout=dropout,
                                            fir=fir,
                                            fir_kernel=fir_kernel,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4,
                                            zemb_dim=z_emb_dim)
            ResnetCondBlock = functools.partial(ResnetBlockBigGAN_cond,
                                                act=act,
                                                dropout=dropout,
                                                fir=fir,
                                                fir_kernel=fir_kernel,
                                                init_scale=init_scale,
                                                skip_rescale=skip_rescale,
                                                temb_dim=nf * 4,
                                                zemb_dim=z_emb_dim)
        elif resblock_type == 'biggan_oneadagn':
            ResnetBlock = functools.partial(ResnetBlockBigGAN_one,
                                            act=act,
                                            dropout=dropout,
                                            fir=fir,
                                            fir_kernel=fir_kernel,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4,
                                            zemb_dim=z_emb_dim)

        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        # Downsampling block

        channels = config.num_channels
        if progressive_input != 'none':
            input_pyramid_ch = channels
            input_pyramid_ch_cond = int(channels * 2)

        modules.append(conv3x3(channels, nf))
        cond_module.append(conv3x3(int(channels * 2), nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                cond_module.append(ResnetCondBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                    cond_module.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                    cond_module.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))
                    cond_module.append(ResnetCondBlock(down=True, in_ch=in_ch))

                if progressive_input == 'input_skip':
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    cond_module.append(combiner(dim1=input_pyramid_ch_cond, dim2=in_ch))
                    if combine_method == 'cat':
                        in_ch *= 2

                elif progressive_input == 'residual':
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    cond_module.append(pyramid_downsample(in_ch=input_pyramid_ch_cond, out_ch=in_ch))
                    input_pyramid_ch_cond = in_ch
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                           out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != 'none':
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name')

            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)
        self.cond_modules = nn.ModuleList(cond_module)

        mapping_layers = [PixelNorm(),
                          dense(config.nz, z_emb_dim),
                          self.act, ]
        for _ in range(config.n_mlp):
            mapping_layers.append(dense(z_emb_dim, z_emb_dim))
            mapping_layers.append(self.act)
        self.z_transform = nn.Sequential(*mapping_layers)

        conv_ch = 256
        self.conv_common1 = nn.Sequential(
            conv3x3(conv_ch, int(conv_ch / 2), padding=1),
            nn.SiLU()
        )
        self.conv_common2 = nn.Sequential(
            conv3x3(conv_ch, int(conv_ch / 2), padding=1),
            nn.SiLU()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se_h1 = nn.Sequential(nn.Linear(conv_ch, conv_ch // 8, bias=False),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(conv_ch // 8, conv_ch, bias=False),
                                   nn.Sigmoid())
        self.se_h2 = nn.Sequential(nn.Linear(conv_ch, conv_ch // 8, bias=False),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(conv_ch // 8, conv_ch, bias=False),
                                   nn.Sigmoid())

        self.dim_reduction_non_zeros = nn.Sequential(
            conv3x3(640, conv_ch, kernel=1, padding=0),
            nn.SiLU()
        )

    def forward(self, x, cond, time_cond, z):
        # timestep/noise_level embedding; only for continuous training
        zemb = self.z_transform(z)
        dec_attentions = []

        modules = self.all_modules
        cond_modules = self.cond_modules

        m_idx = 0
        if self.embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1

        elif self.embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = time_cond

            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.config.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x
            input_pyramid_cond = cond

        hs1 = [modules[m_idx](x)]
        hs2 = [cond_modules[m_idx](cond)]
        hs = [hs2[-1] + hs1[-1]]

        # hs = [hs1[-1]]

        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h1 = modules[m_idx](hs[-1], temb, zemb)
                h2 = cond_modules[m_idx](hs2[-1])
                m_idx += 1
                if h1.shape[-1] in self.attn_resolutions:
                    h1 = modules[m_idx](h1)
                    h2 = cond_modules[m_idx](h2)
                    m_idx += 1

                hs1.append(h1)
                hs2.append(h2)
                hs.append(h1 + h2)


            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h1 = modules[m_idx](hs1[-1])
                    h2 = cond_modules[m_idx](hs2[-1])
                    m_idx += 1
                else:
                    h1 = modules[m_idx](hs1[-1], temb, zemb)
                    h2 = cond_modules[m_idx](hs2[-1])
                    m_idx += 1

                if self.progressive_input == 'input_skip':
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == 'residual':
                    input_pyramid = modules[m_idx](input_pyramid)
                    input_pyramid_cond = cond_modules[m_idx](input_pyramid_cond)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h1) / np.sqrt(2.)
                        input_pyramid_cond = (input_pyramid_cond + h2) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h1
                        input_pyramid_cond = input_pyramid_cond + h2
                    h1 = input_pyramid
                    h2 = input_pyramid_cond

                hs1.append(h1)
                hs2.append(h2)
                hs.append(h1 + h2)

        h1 = hs1[-1]
        h2 = hs2[-1]

        h1_local_feat = self.conv_common1(h1)
        h2_local_feat = self.conv_common2(h2)

        b, c, _, _ = h1.size()

        h1_global_feat = self.avg_pool(h1).view(b, c)
        h1_global_feat = self.se_h1(h1_global_feat).view(b, c, 1, 1)
        h1_global_feat = h1 * h1_global_feat.expand_as(h1)

        h2_global_feat = self.avg_pool(h2).view(b, c)
        h2_global_feat = self.se_h2(h2_global_feat).view(b, c, 1, 1)
        h2_global_feat = h2 * h2_global_feat.expand_as(h2)

        h_local = h1_local_feat + h2_local_feat

        h = torch.cat([h_local, h1_global_feat, h2_global_feat], dim=1)
        h = self.dim_reduction_non_zeros(h)

        h = modules[m_idx](h, temb, zemb)
        m_idx += 1
        h = modules[m_idx](h)
        # mid_att_weight = modules[m_idx].mid_weight.mean(1, keepdim=False)

        # mid_att_weight = F.interpolate(mid_att_weight.unsqueeze(1), (256, 256))
        m_idx += 1
        h = modules[m_idx](h, temb, zemb)
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb, zemb)

                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                dec_att_weight = modules[m_idx].att_weight.mean(1, keepdim=False)
                dec_attentions.append(dec_att_weight)

                m_idx += 1

            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name.')
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name')

            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb, zemb)
                    m_idx += 1

        assert not hs

        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules)

        if not self.not_use_tanh:

            return torch.tanh(h), dec_attentions
        else:
            return h, dec_attentions
