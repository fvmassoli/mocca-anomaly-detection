import sys
from operator import mul
from typing import Tuple
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from mvtec_base_model import BaseModule, DownsampleBlock, ResidualBlock, UpsampleBlock


CHANNELS = [32, 64, 128]


def init_conv_blocks(channel_in: int, channel_out: int, activation_fn: nn):
            """ Init convolutional layers.

            Parameters
            ----------
            k_size : int
                Kernel size
            out_channels : int
                Output features size

            """
            return DownsampleBlock(channel_in=channel_in, channel_out=channel_out, activation_fn=self.activation_fn)


class Selector(nn.Module):
    """Selector module

    """
    def __init__(self, code_length: int, idx: int):
        super().__init__()
        """Init Selector architeture

        Parameters
        ----------
        code_length : int 
            Latent code size
        idx : int
            Layer idx
        
        """
        # List of depths of features maps
        sizes = [CHANNELS[0], CHANNELS[0], CHANNELS[1], CHANNELS[2], CHANNELS[2]*2, CHANNELS[2]*2, code_length]
        
        # Hidden FC output size
        mid_features_size = 256

        # Last FC output size
        out_features = 128

        # Choose a different Selector architecture
        # depending on which layer it attaches
        if idx < 5:
            self.fc = nn.Sequential(
                            nn.AdaptiveMaxPool2d(output_size=8),
                            nn.Conv2d(in_channels=sizes[idx], out_channels=1, kernel_size=1),
                            nn.Flatten(),
                            nn.Linear(in_features=8**2, out_features=mid_features_size, bias=True),
                            nn.BatchNorm1d(mid_features_size),
                            nn.ReLU(),
                            nn.Linear(in_features=mid_features_size, out_features=out_features, bias=True)
                        )
        else:
            self.fc = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(in_features=sizes[idx],  out_features=mid_features_size, bias=True),
                            nn.BatchNorm1d(mid_features_size),
                            nn.ReLU(),
                            nn.Linear(in_features=mid_features_size, out_features=out_features, bias=True)
                        )

    def forward(self, *input: Any):
        return self.fc(input)


class MVtec_Encoder(BaseModule):
    """MVtec Encoder network
    
    """
    def __init__(self, input_shape: torch.Tensor, code_length: int, idx_list_enc: list, use_selectors: bool):
        """Init Encoder network
        
        Parameters
        ----------
        input_shape : torch.Tensor
            Input data shape
        code_length : int
            Latent code size
        idx_list_enc : list 
            List of layers' idx to use for the AD task
        use_selectors : bool
            True (False) if the model has (not) to use Selectors modules

        """
        super().__init__()
        
        self.idx_list_enc = idx_list_enc
        self.use_selectors = use_selectors

        # Single input data shape
        c, h, w = input_shape

        # Activation function
        self.activation_fn = nn.LeakyReLU()

        # Init convolutional blocks
        self.conv = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, bias=False)
        self.res  = ResidualBlock(channel_in=32, channel_out=32, activation_fn=self.activation_fn)
        self.dwn1, self.dwn2, self.dwn3 = [init_conv_blocks(channel_in=ch, channel_out=ch*2, activation_fn=self.activation_fn) for ch in CHANNELS] 
        
        # Depth of the last features map
        self.last_depth = CHANNELS[2]*2

        # Shape of the last features map
        self.deepest_shape = (self.last_depth, h // 8, w // 8) 
        
        # init FC layers
        self.fc1 = nn.Linear(in_features=reduce(mul, self.deepest_shape), out_features=self.last_depth)
        self.bn  = nn.BatchNorm1d(num_features=self.last_depth)
        self.fc2 = nn.Linear(in_features=self.last_depth, out_features=code_length)
        
        ## Init features selector models
        if self.use_selectors:
            self.selectors = nn.ModuleList([Selector(code_length=code_length, idx=idx) for idx in range(7)])
            self.selectors.append(Selector(code_length=code_length, idx=6))

    def get_depths_info(self):
        """
        Return
        ------
        self.last_depth : int
            Depth of the last features map
        self.deepest_shape : int
            Shape of the last features map

        """
        return self.last_depth, self.deepest_shape

    def forward(self, *input: Any):
        o1 = self.conv(input)
        o2 = self.res(self.activation_fn(o1))
        o3 = self.dwn1(o2)
        o4 = self.dwn2(o3)
        o5 = self.dwn3(o4)
        o7 = self.activation_fn(
                            self.bn(
                                self.fc1(
                                    o5.view(len(o5), -1)
                                    )
                                )
                            ) # FC -> BN -> LeakyReLU
        o8 = self.fc2(o7)
        z = nn.Sigmoid()(o8)
        
        outputs = [o1, o2, o3, o4, o5, o7, o8, z]

        if self.use_selectors:
            tuple_o = [self.selectors[idx](tt) for idx, tt in enumerate(outputs) if idx in self.idx_list_enc]
        else:
            tuple_o = []
            for idx, tt in enumerate(outputs):
                if idx not in self.idx_list_enc: continue
                if tt.ndimension() > 2:
                    tuple_o.append(F.avg_pool2d(tt, tt.shape[-2:]).squeeze())
                else:
                    tuple_o.append(tt.squeeze())

        names = [f'0{idx}' for idx in range(len(outputs)) if idx in self.idx_list_enc]
        zipped = list(zip(names, tuple_o))

        return zipped[i] if len(self.idx_list_enc) != 0 else z


class MVTec_Decoder(BaseModule):
    """MVTec Decoder network
    
    """
    def __init__(self, code_length: int, deepest_shape: int, last_depth: int, output_shape: torch.Tensor):
        """Init MVtec Decoder network
        
        Parameters
        ----------
        code_length : int
            Latent code size
        deepest_shape : int
            Depth of the last encoder features map
        output_shape : torch.Tensor
            Input Data shape
        
        """
        super().__init__()

        self.code_length = code_length
        self.deepest_shape = deepest_shape
        self.output_shape = output_shape

        # Decoder activation function
        activation_fn = nn.LeakyReLU()

        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=code_length, out_features=last_depth),
            nn.BatchNorm1d(num_features=last_depth),
            activation_fn,
            nn.Linear(in_features=last_depth, out_features=reduce(mul, deepest_shape)),
            nn.BatchNorm1d(num_features=reduce(mul, deepest_shape)),
            activation_fn
        )

        # (Transposed) Convolutional network
        self.conv = nn.Sequential(
            UpsampleBlock(channel_in=CHANNELS[2]*2, channel_out=CHANNELS[2], activation_fn=activation_fn),
            UpsampleBlock(channel_in=CHANNELS[1]*2, channel_out=CHANNELS[1], activation_fn=activation_fn),
            UpsampleBlock(channel_in=CHANNELS[0]*2, channel_out=CHANNELS[0], activation_fn=activation_fn),
            ResidualBlock(channel_in=CHANNELS[0], channel_out=CHANNELS[0], activation_fn=activation_fn),
            nn.Conv2d(in_channels=CHANNELS[0], out_channels=3, kernel_size=1, bias=False)
        )

    def forward(self, *input: Any):
        h = self.fc(input)
        h = h.view(len(h), *self.deepest_shape)
        return self.conv(h)


class MVTecNet_AutoEncoder(BaseModule):
    """Full MVTecNet_AutoEncoder network
    
    """
    def __init__(self, input_shape: int, code_length: int, use_selectors: bool):
        """Init Full AutoEncoder

        Parameters
        ----------
        input_shape : Tensor
            Shape of input data
        code_length : int
            Latent code size
        use_selectors : bool
            True (False) if the model has (not) to use Selectors modules

        """
        super().__init__()

        # Shape of input data needed by the Decoder
        self.input_shape = input_shape
        
        # Build Encoder
        self.encoder = MVtecEncoder(
                                input_shape=input_shape,
                                code_length=code_length,
                                idx_list_enc=[],
                                use_selectors=use_selectors
                            )

        last_depth, deepest_shape = self.encoder.get_depths_info()

        # Build Decoder
        self.decoder = MVtecDecoder(
                                code_length=code_length,
                                deepest_shape=deepest_shape,
                                last_depth=last_depth,
                                output_shape=input_shape
                            )

    def forward(self, *input: Any):
        z = self.encoder(input)
        x_r = self.decoder(z)
        x_r = x_r.view(-1, *self.input_shape)
        return x_r
