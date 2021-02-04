import os
import sys
import random
import logging
import argparse
import numpy as np

import torch

from tensorboardX import SummaryWriter

from datasets.data_manager import DataManager
from trainers.train_cifar10 import pretrain, train, test
from utils import set_seeds, get_out_dir, purge_ae_params
from models.cifar10_model import CIFAR10_Autoencoder, CIFAR10_Encoder


def main(args):

    # If the layer list is not specified, them use only the last layer to detect anomalies
    if len(args.idx_list_enc) == 0 and args.train:
        args.idx_list_enc = [3]

    ## Init logger & print training/warm-up summary
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler('./training.log'),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()
    if args.train or args.pretrain:
        logger.info(
                "Start run with params:"
                f"\n\t\t\t\tPretrain model   : {args.pretrain}"
                f"\n\t\t\t\tTrain model      : {args.train}"
                f"\n\t\t\t\tTest model       : {args.test}"
                f"\n\t\t\t\tBoundary         : {args.boundary}"
                f"\n\t\t\t\tDataset          : {args.dataset_name}"
                f"\n\t\t\t\tUse unl data     : {args.unlabelled_data}"
                f"\n\t\t\t\tNormal class     : {args.normal_class}"
                f"\n\t\t\t\tBatch size       : {args.batch_size}\n"
                f"\n\t\t\t\tOptimizer        : {args.optimizer}"
                f"\n\t\t\t\tPretrain epochs  : {args.ae_epochs}"
                f"\n\t\t\t\tAE-Learning rate : {args.ae_learning_rate}"
                f"\n\t\t\t\tAE-milestones    : {args.ae_lr_milestones}"
                f"\n\t\t\t\tAE-Weight decay  : {args.ae_weight_decay}\n"
                f"\n\t\t\t\tTrain epochs     : {args.epochs}"
                f"\n\t\t\t\tLearning rate    : {args.learning_rate}"
                f"\n\t\t\t\tMilestones       : {args.lr_milestones}"
                f"\n\t\t\t\tWeight decay     : {args.weight_decay}\n"
                f"\n\t\t\t\tCode length      : {args.code_length}"
                f"\n\t\t\t\tNu               : {args.nu}"
                f"\n\t\t\t\tEncoder list     : {args.idx_list_enc}\n"
            )
    else:
        if args.model_ckp is None:
            logger.info("CANNOT TEST MODEL WITHOUT A VALID CHECKPOINT")
            sys.exit(0)
        args.normal_class = int(args.model_ckp.split('/')[-2].split('-')[2].split('_')[-1])

    # Set seed
    set_seeds(args.seed)

    # Get the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Init DataHolder class
    data_holder = DataManager(
                        dataset_name=args.dataset_name, 
                        data_path=args.data_path, 
                        normal_class=args.normal_class, 
                        only_test=args.test
                    ).get_data_holder()

    # Load data
    train_loader, test_loader = data_holder.get_loaders(
                                                    batch_size=args.batch_szie, 
                                                    shuffle_train=True, 
                                                    pin_memory=device=="cuda", 
                                                    num_workers=args.n_workers
                                                )
    
    ### PRETRAIN the full AutoEncoder
    ae_net_cehckpoint = None
    if args.pretrain:        
        out_dir, tmp = get_out_dir(args, pretrain=True, aelr=None, net_name='cifar10')
        tb_writer = SummaryWriter(os.path.join(args.output_path, args.dataset_name, str(args.normal_class), 'svdd/tb_runs_pretrain', tmp))
        
        # Init AutoEncoder
        ae_net = CIFAR10_Autoencoder(args.code_length)
        
        # Start pretraining
        ae_net_cehckpoint = pretrain(ae_net, train_loader, out_dir, tb_writer, device, args)
        tb_writer.close()

    ### TRAIN the Encoder
    net_cehckpoint = None
    if args.train:
        if ae_net_cehckpoint is None:
            if args.model_ckp is None:
                logger.info("CANNOT TRAIN MODEL WITHOUT A VALID CHECKPOINT")
                sys.exit(0)
            ae_net_cehckpoint = args.model_ckp
        aelr = float(ae_net_cehckpoint.split('/')[-2].split('-')[4].split('_')[-1])
        out_dir, tmp = get_out_dir(args, pretrain=False, aelr=aelr)
        
        tb_writer = SummaryWriter(os.path.join(args.output_path, args.dataset_name, str(args.normal_class), 'cifar10/tb_runs_train', tmp))
        
        # Init Encoder
        encoder_net = CIFAR10_Encoder(args.code_length)

        # Load Encoder parameters from pretrianed full AutoEncoder
        purge_ae_params(encoder_net=encoder_net, ae_net_cehckpoint=ae_net_cehckpoint)

        # Start training
        net_cehckpoint = train(encoder_net, train_loader, out_dir, tb_writer, device, ae_net_cehckpoint, args)
        tb_writer.close()

    ### TEST the Encoder
    if args.test:   
        if net_cehckpoint is None:
            net_cehckpoint = args.model_ckp
        # Init Encoder
        net = CIFAR10_Encoder(args.code_length)
        st_dict = torch.load(net_cehckpoint)
        net.load_state_dict(st_dict['net_state_dict'])
        
        logger.info(f"Loaded model from: {net_cehckpoint}")
        if args.debug:
            idx_list_enc = args.idx_list_enc
            boundary = args.boundary
        else:
            idx_list_enc = [int(i) for i in net_cehckpoint.split('/')[-2].split('-')[-1].split('_')[-1].split('.')]
            boundary = net_cehckpoint.split('/')[-2].split('-')[-3].split('_')[-1]
        logger.info(
            f"Start test with params"
            f"\n\t\t\t\tCode length    : {args.code_length}"
            f"\n\t\t\t\tEnc layer list : {idx_list_enc}"
            f"\n\t\t\t\tBoundary       : {boundary}"
            f"\n\t\t\t\tNormal class   : {args.normal_class}"
        )

        # Start test
        test(net, test_loader, st_dict['R'], st_dict['c'], device, idx_list_enc, boundary, args)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('AD')
    ## General config
    parser.add_argument('-s', '--seed', type=int, default=-1, help='Random seed (default: -1)')
    parser.add_argument('--n_workers', type=int, default=8, help='Number of workers for data loading. 0 means that the data will be loaded in the main process. (default: 8)')
    parser.add_argument('--output_path', default='./output/cifar10_ad')
    ## Model config
    parser.add_argument('-zl', '--code-length', default=32, type=int, help='Code length (default: 32)')
    parser.add_argument('-ck', '--model-ckp', help='Model checkpoint')
    ## Optimizer config
    parser.add_argument('-alr', '--ae-learning-rate', type=float, default=1.e-4, help='Warm up learning rate (default: 1.e-4)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1.e-4, help='Learning rate (default: 1.e-4)')
    parser.add_argument('-awd', '--ae-weight-decay', type=float, default=0.5e-6, help='Warm up learning rate (default: 0.5e-4)')
    parser.add_argument('-wd', '--weight-decay', type=float, default=0.5e-6, help='Learning rate (default: 0.5e-6)')
    parser.add_argument('-aml', '--ae-lr-milestones', type=int, nargs='+', default=[], help='Pretrain milestone')
    parser.add_argument('-ml', '--lr-milestones', type=int, nargs='+', default=[], help='Training milestone')
    ## Data
    parser.add_argument('-dp', '--data-path', help='Dataset main path')
    parser.add_argument('-nc', '--normal-class', type=int, default=5, help='Normal Class (default: 5)')
    ## Training config
    parser.add_argument('-we', '--warm_up_n_epochs', type=int, default=10, help='Warm up epochs (default: 10)')
    parser.add_argument('--use-selectors', action="store_true", help='Use features selector (default: False)')
    parser.add_argument('-tbc', '--train-best-conf', action="store_true", help='Train best configurations (default: False)')
    parser.add_argument('-db', '--debug', action="store_true", help='Debug (default: False)')
    parser.add_argument('-bs', '--batch-size', type=int, default=200, help='Batch size (default: 200)')
    parser.add_argument('-bd', '--boundary', choices=("hard", "soft"), default="soft", help='Boundary (default: soft)')
    parser.add_argument('-ptr', '--pretrain', action="store_true", help='Pretrain model (default: False)')
    parser.add_argument('-tr', '--train', action="store_true", help='Train model (default: False)')
    parser.add_argument('-tt', '--test', action="store_true", help='Test model (default: False)')
    parser.add_argument('-ile', '--idx-list-enc', type=int, nargs='+', default=[], help='List of indices of model encoder')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Training epochs (default: 1)')
    parser.add_argument('-ae', '--ae-epochs', type=int, default=1, help='Warmp up epochs (default: 1)')
    parser.add_argument('-nu', '--nu', type=float, default=0.1)
    args = parser.parse_args()

    main(args)
