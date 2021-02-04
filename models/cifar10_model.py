import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_conv(out_channels: int, k_size: int = 5):
    """ Init convolutional layers.

    Parameters
    ----------
    k_size : int
        Kernel size
    out_channels : int
        Output features size

    Returns
    -------
    nn.Module : 
        Conv2d layer

    """
    l = nn.Conv2d(
            in_channels=3 if out_channels==32 else out_channels//2, 
            out_channels=out_channels, 
            kernel_size=k_size, 
            bias=False, 
            padding=2
        )
    nn.init.xavier_uniform_(l.weight, gain=nn.init.calculate_gain('leaky_relu'))
    return l
    

def init_deconv(out_channels: int, k_size: int = 5):
    """ Init deconv layers.

    Parameters
    ----------
    k_size : int
        Kernel size
    out_channels : int
        Input features size

    Returns
    -------
    nn.Module : 
        ConvTranspose2d layer

    """
    l = nn.ConvTranspose2d(
            in_channels=out_channels, 
            out_channels=3 if out_channels==32 else out_channels//2, 
            kernel_size=k_size, 
            bias=False, 
            padding=2
        )
    nn.init.xavier_uniform_(l.weight, gain=nn.init.calculate_gain('leaky_relu'))
    return l


class BaseNet(nn.Module):
    """Base class for all neural networks.
    
    """
    def __init__(self):
        super().__init__()

        # init Logger to print model infos
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # List of input/output features depths for the convolutional layers
        self.output_features_sizes = [32, 64, 128]
    
    def __init_bn(self, num_features: int):
            """ Init BatchNorm layers.

            Parameters
            ----------
            num_features : int
                Number of input features

            """
            return nn.BatchNorm2d(num_features=num_features, eps=1e-04, affine=False)

    def summary(self):
        """Network summary.
        
        """
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)


class CIFAR10_Encoder(BaseNet):
    """"Encoder network.
    
    """
    def __init__(self, code_length: int):
        """"Init encoder.

        Parameters
        ----------
        code_length : int
            Latent code size
        
        """
        super().__init__()
        
        # Init Conv layers
        self.conv1, self.conv2, self.conv3 = [init_conv(out_channels) for out_channels in self.output_features_sizes]
        
        # Init BN layers
        self.bnd1, self.bnd2, self.bnd3 = [self.__init_bn(num_features) for num_features in self.output_features_sizes]

        # Init all other layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=code_length, bias=False)
    
    def forward(self, *input: Any):
        x = self.conv1(input)
        x = self.pool(F.leaky_relu(self.bnd1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bnd2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bnd3(x)))
        x = self.fc1(x.view(x.size(0), -1))
        return x


class CIFAR10_Decoder(BaseNet):
    """Full Decoder network.

    """ 
    def __init__(self, code_length: int):
        """Init decoder.

        Parameters
        ----------
        code_length : int
            Latent code size

        """
        super().__init__()

        self.rep_dim = code_length
        
        # Build the Decoder
        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 128, 5, bias=False, padding=2)
        self.deconv2, self.deconv3, self.deconv4 = [init_deconv(out_channels) for out_channels in self.output_features_sizes[::-1]]

        # Init BN layers
        self.bnd4, self.bnd5, self.bnd6 = [self.__init_bn(num_features) for num_features in self.output_features_sizes[::-1]] 

    def forward(self, *input: Any):
        x = self.bn1d(input)
        x = x.view(x.size(0), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bnd4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bnd5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bnd6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x


class CIFAR10_Autoencoder(BaseNet):
    """Full AutoEncoder network.

    """ 
    def __init__(self, code_length: int = 128):
        """Init the AutoEncoder 

        Parameters
        ----------
        code_length : int
            Latent code size
        
        """
        super().__init__()
        
        # Build the Encoder
        self.encoder = CIFAR10_Encoder(code_length=code_length)
        self.bn1d = nn.BatchNorm1d(num_features=code_length, eps=1e-04, affine=False)

        # Build the Decoder
        self.decoder = CIFAR10_Decoder(code_length=code_length)
        
    def forward(self, *input: Any):
        z = self.encoder(input)
        z = self.bn1d(z)
        return self.decoder(z)
