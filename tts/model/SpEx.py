from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tts.base import BaseModel


class GlobalLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(dim, 1))
            self.gamma = nn.Parameter(torch.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} requires a 3D tensor input".format(
                self.__name__))
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class Normalization(nn.Module):
    _torch_unlinear = {
        'ReLU': F.relu,
        'ReLU6': F.relu6,
        'LeakyReLU': F.leaky_relu,
        'PReLU': F.prelu
    }

    def __init__(self, input_dim: int, unlinear: str):
        super().__init__()
        if unlinear == 'PReLU':
            self.unlinear = nn.PReLU()
        else:
            self.unlinear = self._torch_unlinear[unlinear]
        self.normalization = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor):
        x = torch.transpose(x, 1, 2)
        ans = self.unlinear(self.normalization(x))
        return torch.transpose(ans, 1, 2)


class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 L1: int,
                 L2: int,
                 L3: int):
        super(Encoder, self).__init__()
        self.kernels = [L1, L2, L3]
        self.short = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=L1, stride=L1 // 2)
        self.middle = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=L2, stride=L1 // 2)
        self.long = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=L3, stride=L1 // 2)

    def forward(self, x):
        if x.shape != 3:
            x = torch.unsqueeze(x, 1)
        short = self.short(x)
        middle_shape = (short.shape[-1] - 1) * (self.kernels[0] // 2) + self.kernels[1]
        long_shape = (short.shape[-1] - 1) * (self.kernels[0] // 2) + self.kernels[2]
        middle = self.middle(F.pad(x, (0, middle_shape - x.shape[-1])))
        long = self.long(F.pad(x, (0, long_shape - x.shape[-1])))
        return short, middle, long, torch.cat([short, middle, long], 1)


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        if in_channels != out_channels:
            self.downcample = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        else:
            self.downcample = None
        self.final_prelu = nn.PReLU()

    def forward(self, x):
        block = self.block(x)
        if self.downcample:
            block += self.downcample(x)
        else:
            block += x
        return self.final_prelu(block)


class TCN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 spk_embed_dim: int,
                 conv_channels: int,
                 kernel_size: int,
                 dilation: int):
        super(TCN, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels + spk_embed_dim, conv_channels, 1),
            nn.PReLU(),
            GlobalLayerNorm(conv_channels, elementwise_affine=True),
            nn.Conv1d(
                conv_channels,
                conv_channels,
                kernel_size,
                groups=conv_channels,
                padding=(dilation * (kernel_size - 1)) // 2,
                dilation=dilation,
                bias=True),
            nn.PReLU(),
            GlobalLayerNorm(conv_channels, elementwise_affine=True),
            nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        )

    def forward(self, x, ref=None):
        if ref is not None:

            aux = torch.unsqueeze(ref, -1).repeat(1, 1, x.shape[-1])
            y = torch.cat([x, aux], 1)
        else:
            y = x
        y = self.block(y)
        return x + y


class TCN_block(nn.Module):
    def __init__(self,
                 in_channels: int,
                 spk_embed_dim: int,
                 conv_channels: int,
                 kernel_size: int,
                 len_of_chain: int):
        super(TCN_block, self).__init__()
        self.pipeline = nn.ModuleList(
            [TCN(in_channels=in_channels,
                 spk_embed_dim=spk_embed_dim * int(i == 0),
                 conv_channels=conv_channels,
                 kernel_size=kernel_size,
                 dilation=2 ** i
                 ) for i in range(len_of_chain)]
        )

    def forward(self, x, ref):
        for ind, module in enumerate(self.pipeline):
            x = module(x, ref) if not ind else module(x)
        return x


class SpEx(BaseModel):
    _torch_unlinear = {
        'ReLU': F.relu,
        'ReLU6': F.relu6,
        'LeakyReLU': F.leaky_relu,
        'PReLU': F.prelu
    }

    def __init__(self, n_feats,
                 n_class,
                 encoder_params,
                 norm_params,
                 tcn_params,
                 resnet_count,
                 final_params):
        super(SpEx, self).__init__(n_feats, n_class)
        self.L = [encoder_params.L1, encoder_params.L2, encoder_params.L3]
        self.encoder = Encoder(in_channels=1,
                               out_channels=encoder_params.out_channels,
                               L1=encoder_params.L1,
                               L2=encoder_params.L2,
                               L3=encoder_params.L3)

        # Create left_path from encoder to start of Stacked	TCNs
        self.mix_norm = nn.Sequential(
            Normalization(3 * encoder_params.out_channels, norm_params.unlinear),
            nn.Conv1d(in_channels=3 * encoder_params.out_channels,
                      out_channels=norm_params.out_channels,
                      kernel_size=1)
        )

        # Create Stacked TCNs blocks
        self.tcn = nn.ModuleList(
            [TCN_block(
                in_channels=norm_params.out_channels,
                spk_embed_dim=tcn_params.spk_embed_dim,
                conv_channels=tcn_params.conv_channels,
                kernel_size=tcn_params.kernel_size,
                len_of_chain=tcn_params.len_of_tcn_chain,
            ) for _ in range(tcn_params.count_of_tch)]
        )

        # Create right_path from encoder to end of ResNet
        self.ref_net = nn.Sequential(
            Normalization(3 * encoder_params.out_channels, norm_params.unlinear),
            nn.Conv1d(in_channels=3 * encoder_params.out_channels,
                      out_channels=norm_params.out_channels,
                      kernel_size=1)
        )
        for i in range(resnet_count):
            in_channels = norm_params.out_channels if i * 2 <= resnet_count else final_params.out_channels
            out_channels = norm_params.out_channels if i * 2 <= resnet_count - 2 else final_params.out_channels
            self.ref_net.append(
                ResNet(in_channels=in_channels, out_channels=out_channels)
            )

        # Create end of left had
        self.mask = nn.ModuleList(
            [
                nn.Conv1d(norm_params.out_channels, encoder_params.out_channels, 1)
                for _ in range(3)
            ]
        )
        if final_params.unlinear == 'PReLU':
            self.mask_unlinear = nn.PReLU()
        else:
            self.mask_unlinear = self._torch_unlinear[final_params.unlinear]
        self.decoder_short = nn.ConvTranspose1d(in_channels=encoder_params.out_channels, out_channels=1,
                                                kernel_size=encoder_params.L1, stride=encoder_params.L1 // 2)
        self.decoder_middle = nn.ConvTranspose1d(in_channels=encoder_params.out_channels, out_channels=1,
                                                 kernel_size=encoder_params.L2, stride=encoder_params.L1 // 2)
        self.decoder_long = nn.ConvTranspose1d(in_channels=encoder_params.out_channels, out_channels=1,
                                               kernel_size=encoder_params.L3, stride=encoder_params.L1 // 2)

        # Create end of right hand
        self.right_hand = nn.Linear(
            final_params.out_channels,
            n_class
        )

    def forward(self, **batch):
        print(batch.keys())
        mixed = batch['mixed']
        short_mix, middle_mix, long_mix, enc_mix = self.encoder(mixed)
        enc_mix = self.mix_norm(enc_mix)

        ref = batch['ref']
        _, _, _, enc_ref = self.encoder(ref)
        ref = self.ref_net(enc_ref)

        ref_T = (batch['ref'].shape[-1] - self.L[0]) // (self.L[0] // 2) + 1
        ref_T = ((ref_T // 3) // 3) // 3
        ref = torch.sum(ref, dim=-1) / float(ref_T)

        for module in self.tcn:
            enc_mix = module(enc_mix, ref)

        mask_short = self.mask_unlinear(self.mask[0](enc_mix))
        short_pred = self.decoder_short(mask_short * short_mix).squeeze(1)
        short_pred = F.pad(short_pred, [0, batch['mixed'].shape[-1] - short_pred.shape[-1]])

        mask_middle = self.mask_unlinear(self.mask[1](enc_mix))
        middle_pred = self.decoder_middle(mask_middle * middle_mix).squeeze(1)[:, :batch['mixed'].shape[-1]]

        mask_long = self.mask_unlinear(self.mask[2](enc_mix))
        long_pred = self.decoder_long(mask_long * long_mix).squeeze(1)[:, :batch['mixed'].shape[-1]]

        right_head = self.right_hand(ref)

        return {"short": short_pred, "middle": middle_pred, "long": long_pred, "logits": right_head}

    def transform_input_lengths(self, input_lengths):
        return input_lengths
