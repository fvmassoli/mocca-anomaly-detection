import os
import random
import logging
import numpy as np
from tqdm import tqdm
from PIL import Image
from os.path import join

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from models.mvtec_model import MVtec_Encoder


def get_out_dir(args, pretrain: bool, aelr: float, net_name: str="cifar10", training_strategy: str=None):
    """Creates training output dir

    Parameters
    ----------

    args : 
        Arguments
    pretrain : bool
        True if pretrain the model
    aelr : float
        Full AutoEncoder learning rate
    net_name : str
        Netwrok name
    training_strategy : str
        ................................................................
    
    """
    if pretrain:
        tmp = (f"pretrain-mn_{net_name}-nc_{args.normal_class}-cl_{args.code_length}-lr_{args.ae_learning_rate}-unl_{args.unlabelled_data}-awd_{args.ae_weight_decay}")
        if training_strategy is not None:
            tmp = (f"{training_strategy}-pretrain-mn_{net_name}-nc_{args.normal_class}-cl_{args.code_length}-lr_{args.ae_learning_rate}-unl_{args.unlabelled_data}")
        out_dir = os.path.join(args.output_path, args.dataset_name, str(args.normal_class), net_name, 'pretrain', tmp)
    else:
        tmp = (
            f"train-mn_{net_name}-nc_{args.normal_class}-cl_{args.code_length}-bs_{args.batch_size}-nu_{args.nu}-lr_{args.learning_rate}-"
            f"wd_{args.weight_decay}-bd_{args.boundary}-unl_{args.unlabelled_data}-alr_{aelr}-sl_{args.use_selectors}-ep_{args.epochs}-ile_{'.'.join(map(str, args.idx_list_enc))}"
        )
        if training_strategy is not None:
            tmp = (
                f"{training_strategy}-train-mn_{net_name}-nc_{args.normal_class}-cl_{args.code_length}-bs_{args.batch_size}-nu_{args.nu}-lr_{args.learning_rate}-"
                f"wd_{args.weight_decay}-bd_{args.boundary}-unl_{args.unlabelled_data}-alr_{aelr}-sl_{args.use_selectors}-ep_{args.epochs}-ile_{'.'.join(map(str, args.idx_list_enc))}"
            )
        str__ = 'train_best_conf' if args.train_best_conf else 'train'
        out_dir = os.path.join(args.output_path, args.dataset_name, str(args.normal_class), net_name, str__, tmp)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir, tmp


def set_seeds(seed: int):
    """Set all seeds.
    
    Parameters
    ----------
    seed : int
        Seed

    """
    # Set the seed only if the user specified it
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def purge_ae_params(encoder_net, ae_net_cehckpoint: str):
    """Load Encoder preatrained weights from the full AutoEncoder.
    After the pretraining phase, we don't need the full AutoEncoder parameters, we only need the Encoder
    
    Parameters
    ----------
    encoder_net :
        The Encoder network
    ae_net_cehckpoint : str
        Path to full AutoEncoder checkpoint
    
    """
    # Load the full AutoEncoder checkpoint dict
    ae_net_dict = torch.load(ae_net_cehckpoint, map_location=lambda storage, loc: storage)['ae_state_dict']
        
    # Load encoder weight from autoencoder
    net_dict = encoder_net.state_dict()
    
    # Filter out decoder network keys
    st_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
    
    # Overwrite values in the existing state_dict
    net_dict.update(st_dict)
    
    # Load the new state_dict
    encoder_net.load_state_dict(net_dict)
        

def load_mvtec_model_from_checkpoint(input_shape: tuple, code_length: int, idx_list_enc: list, use_selectors: bool, net_cehckpoint: str, purge_ae_params: bool = False):
    """Load AutoEncoder checkpoint. 
    
    Parameters
    ----------
    input_shape : tuple
        Input data shape
    code_length : int
        Latent code size
    idx_list_enc : list
        List of indexes of layers from which extract features
    use_selectors : bool
        True if the model has to use Selector modules
    net_cehckpoint : str
        Path to model checkpoint
    purge_ae_params : bool 
        True if the checkpoint is relative to an AutoEncoder

    Returns
    -------
    encoder_net : nn.Module
        The Encoder network

    """
    logger = logging.getLogger()

    encoder_net = MVtec_Encoder(
                            input_shape=input_shape,
                            code_length=code_length,
                            idx_list_enc=idx_list_enc,
                            use_selectors=use_selectors
                        )
    
    if purge_ae_params:
    
        # Load Encoder parameters from pretrianed full AutoEncoder
        logger.info(f"Loading encoder from: {net_cehckpoint}")
        purge_ae_params(encoder_net=encoder_net, ae_net_cehckpoint=net_cehckpoint)
    else:

        st_dict = torch.load(net_cehckpoint)
        encoder_net.load_state_dict(st_dict['net_state_dict'])
        logger.info(f"Loaded model from: {net_cehckpoint}")
    
    return encoder_net


def eval_spheres_centers(train_loader: DataLoader, encoder_net: torch.nn.Module, ae_net_cehckpoint: str, debug: bool):
    """Eval the centers of the hyperspheres at each chosen layer.

    Parameters
    ----------
    train_loader : DataLoader
        DataLoader for trainin data
    encoder_net : torch.nn.Module
        Encoder network 
    ae_net_cehckpoint : str
        Checkpoint of the full AutoEncoder 
    debug : bool
        Activate debug mode
    
    Returns
    -------
    dict : dictionary
        Dictionary with k='layer name'; v='features vector representing hypersphere center' 
    
    """
    logger = logging.getLogger()
    # If centers are found, then load and return
    if os.path.exists(ae_net_cehckpoint[:-4]+'_w_centers.pth'):
        logger.info("Found hyperspheres centers")
        ae_net_ckp = torch.load(ae_net_cehckpoint[:-4]+'_w_centers.pth', map_location=lambda storage, loc: storage)
        centers = {k: v.to(device) for k, v in ae_net_ckp['centers'].items()}
    else:
        logger.info("Hyperspheres centers not found... evaluating...")
        centers_ = init_center_c(train_loader=train_loader, encoder_net=encoder_net, debug=debug)
        logger.info("Hyperspheres centers evaluated!!!")
        new_ckp = ae_net_cehckpoint.split('.pth')[0]+'_w_centers.pth'
        logger.info(f"New AE dict saved at: {new_ckp}!!!")
        centers = {k: v for k, v in centers_.items()}
        torch.save({
                'ae_state_dict': ae_net_ckp['ae_state_dict'],
                'centers': centers
                }, new_ckp)
    return centers


@torch.no_grad()
def init_center_c(train_loader: DataLoader, encoder_net: torch.nn.Module, debug: bool, eps: float=0.1):
    """Initialize hypersphere center as the mean from an initial forward pass on the data.
    
    Parameters
    ----------
    train_loader : 
    encoder_net : 
    debug : 
    eps: 

    Returns
    -------
    dictionary : dict
        Dictionary with k='layer name'; v='center featrues'

    """
    n_samples = 0
    net.eval()
    for idx, (data, _) in enumerate(tqdm(train_loader, desc='Init hyperspheres centeres', total=len(train_loader), leave=False)):
        if debug and idx == 2: break
        # get the inputs of the batch
        if isinstance(data, list): data = data[0]
        data = data.to(device)
        n_samples += data.shape[0]
        zipped = net(data)
        if isinstance(zipped, torch.Tensor):
            zipped = [('08', zipped)]
        
        if idx == 0:
            c = {item[0]: torch.zeros_like(item[1][-1], device=device) for item in zipped}
        for item in zipped:
            c[item[0]] += torch.sum(item[1], dim=0)
    
    for k in c.keys():
        c[k] = c[k] / n_samples
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[k][(abs(c[k]) < eps) & (c[k] < 0)] = -eps
        c[k][(abs(c[k]) < eps) & (c[k] > 0)] = eps

    return c
