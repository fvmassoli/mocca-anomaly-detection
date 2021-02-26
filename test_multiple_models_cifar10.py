import os
import sys
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch

from trainer_svdd import test
from datasets.main import load_dataset
from models.deep_svdd.deep_svdd_mnist import MNIST_LeNet_Autoencoder, MNIST_LeNet
from models.deep_svdd.deep_svdd_cifar10 import CIFAR10_LeNet_Autoencoder, CIFAR10_LeNet


parser = argparse.ArgumentParser('AD')
## General config
parser.add_argument('--n_jobs_dataloader', type=int, default=0, help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
## Model config
parser.add_argument('-zl', '--code-length', default=32, type=int, help='Code length (default: 32)')
parser.add_argument('-ck', '--model-ckp', help='Model checkpoint')
## Data
parser.add_argument('-ax', '--aux-data-filename', default='/media/fabiovalerio/datasets/ti_500K_pseudo_labeled.pickle', help='Path to unlabelled data')
parser.add_argument('-dn', '--dataset-name', choices=('mnist', 'cifar10'), default='mnist', help='Dataset (default: mnist)')
parser.add_argument('-ul', '--unlabelled-data', action="store_true", help='Use unlabelled data (default: False)')
parser.add_argument('-aux', '--unl-data-path', default="/media/fabiovalerio/datasets/ti_500K_pseudo_labeled.pickle", help='Path to unalbelled data')
## Training config
parser.add_argument('-bs', '--batch-size', type=int, default=200, help='Batch size (default: 200)')
parser.add_argument('-bd', '--boundary', choices=("hard", "soft"), default="soft", help='Boundary (default: soft)')
parser.add_argument('-ile', '--idx-list-enc', type=int, nargs='+', default=[], help='List of indices of model encoder')
args = parser.parse_args()


# Get data base path
_user = os.environ['USER']
if _user == 'fabiovalerio':
    data_path = '/media/fabiovalerio/datasets'
elif _user == 'fabiom':
    data_path = '/mnt/datone/datasets/'
else:
    raise NotImplementedError('Username %s not configured' % _user)


def main():
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')

    boundary = args.model_ckp.split('/')[-1].split('-')[-3].split('_')[-1]
    normal_class = int(args.model_ckp.split('/')[-1].split('-')[2].split('_')[-1])
    if len(args.idx_list_enc) == 0:
        idx_list_enc = [int(i) for i in args.model_ckp.split('/')[-1].split('-')[-1].split('_')[-1].split('.')]
    else:
        idx_list_enc = args.idx_list_enc
    
    # LOAD DATA
    dataset = load_dataset(args.dataset_name, data_path, normal_class, args.unlabelled_data, args.unl_data_path)

    print(
        f"Start test with params"
        f"\n\t\t\t\tCode length    : {args.code_length}"
        f"\n\t\t\t\tEnc layer list : {idx_list_enc}"
        f"\n\t\t\t\tBoundary       : {boundary}"
        f"\n\t\t\t\tNormal class   : {normal_class}"
    )

    test_auc = []
    main_model_ckp_dir = args.model_ckp
    for m_ckp in tqdm(os.listdir(main_model_ckp_dir), total=len(os.listdir(main_model_ckp_dir)), leave=False):
        net_cehckpoint = os.path.join(main_model_ckp_dir, m_ckp)

        # Load model
        net = MNIST_LeNet(args.code_length) if args.dataset_name == 'mnist' else CIFAR10_LeNet(args.code_length)
        st_dict = torch.load(net_cehckpoint)
        net.load_state_dict(st_dict['net_state_dict'])
        
        # TEST
        test_auc_ = test(net, dataset, st_dict['R'], st_dict['c'], device, idx_list_enc, boundary, args)
        del net, st_dict

        test_auc.append(test_auc_)

    test_auc = np.asarray(test_auc)
    test_auc_m, test_auc_s = test_auc.mean(), test_auc.std()
    print("[")
    for tau in test_auc:
        print(tau, ", ")
    print("]")
    print(test_auc)
    print(f"{test_auc_m:.2f} $\pm$ {test_auc_s:.2f}")


if __name__ == '__main__':
    main()
