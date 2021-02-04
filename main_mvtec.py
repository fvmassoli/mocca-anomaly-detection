import os
import sys
import glob
import random
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from prettytable import PrettyTable

import torch
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from datasets.data_manager import DataManager
from models.mvtec_model import MVTecNet_AutoEncoder
from trainers.train_mvtec import pretrain, train, test
from utils import set_seeds, get_out_dir, purge_ae_params, eval_spheres_centers, load_mvtec_model_from_checkpoint


def test_models(test_loader: DataLoader, net_cehckpoint: str, tables: tuple, out_df: pd.DataFrame, is_texture: bool, input_shape: tuple, idx_list_enc: list, boundary: str, normal_class: str, use_selectors: bool, device: str, debug: bool):
    """Test a single model.
    
    Parameters
    ----------
    test_loader : DataLoader
        Test data loader
    
    net_cehckpoint : str
        Path to model checkpoint
    tables : tuple
        Tuple containing PrettyTabels for soft and hard boundary
    out_df : DataFrame
        Output dataframe
    is_texture : bool
        True if we are dealing with texture-type class
    input_shape : tuple
        Shape of the input data
    idx_list_enc : list
        List containing the index of layers from which extract features
    boundary : str
        Type of boundary
    normal_class : str
        Name of the normal class
    use_selectors : bool
        True if we want to use Selector modules
    device : str
        Device to be used
    debug : bool
        If True, set the debug mode

    Returns
    -------
    out_df : DataFrame
        Dataframe containing the test results

    """
    logger = logging.getLogger()

    if not os.path.exists(net_cehckpoint):
        print(f"File not found at: {net_cehckpoint}")
        return out_df

    # Get latent code size from checkpoint name
    code_length = int(net_cehckpoint.split('/')[-2].split('-')[3].split('_')[-1])
    
    if not debug:
        # If not in debug mode we read the values from the model checkpoint, 
        # otherwise, i.e., in debug mode, read the values from args
        if net_cehckpoint.split('/')[-2].split('-')[-1].split('_')[-1].split('.')[0] == '':
            idx_list_enc = [7]
        idx_list_enc = [int(i) for i in net_cehckpoint.split('/')[-2].split('-')[-1].split('_')[-1].split('.')]
        boundary = net_cehckpoint.split('/')[-2].split('-')[9].split('_')[-1]
        normal_class = net_cehckpoint.split('/')[-2].split('-')[2].split('_')[-1]
    
    logger.info(
        f"Start test with params"
        f"\n\t\t\t\tCode length    : {code_length}"
        f"\n\t\t\t\tEnc layer list : {idx_list_enc}"
        f"\n\t\t\t\tBoundary       : {boundary}"
        f"\n\t\t\t\tObject class   : {normal_class}"
    )

    # Init Encoder
    net = load_mvtec_model_from_checkpoint(
                                    input_shape=input_shape, 
                                    code_length=code_length, 
                                    idx_list_enc=idx_list_enc, 
                                    use_selectors=use_selectors, 
                                    net_cehckpoint=net_cehckpoint
                                )

    ### TEST
    test_auc, test_b_acc = test(normal_class, texture, net, test_dset, st_dict['R'], st_dict['c'], device, idx_list_enc, boundary, args)
    
    table = tables[0] if boundary == 'soft' else tables[1]
    table.add_row([
                net_cehckpoint.split('/')[-2],
                code_length,
                idx_list_enc,
                net_cehckpoint.split('/')[-2].split('-')[7].split('_')[-1]+'-'+net_cehckpoint.split('/')[-2].split('-')[8],
                normal_class,
                boundary,
                net_cehckpoint.split('/')[-2].split('-')[4].split('_')[-1],
                net_cehckpoint.split('/')[-2].split('-')[5].split('_')[-1],
                test_auc,
                test_b_acc
            ])

    out_df = out_df.append(dict(
                            path=net_cehckpoint.split('/')[-2],
                            code_length=code_length,
                            enc_l_list=idx_list_enc,
                            weight_decay=net_cehckpoint.split('/')[-2].split('-')[7].split('_')[-1]+'-'+net_cehckpoint.split('/')[-2].split('-')[8],
                            object_class=normal_class,
                            boundary=boundary,
                            batch_size=net_cehckpoint.split('/')[-2].split('-')[4].split('_')[-1],
                            nu=net_cehckpoint.split('/')[-2].split('-')[5].split('_')[-1],
                            auc=test_auc,
                            balanced_acc=test_b_acc
                        ),
                      ignore_index=True
                  )

    return out_df


def main(args):
    if args.disable_logging:
        logging.disable(level=logging.INFO)

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
                f"\n\t\t\t\tOptimizer        : {args.optimizer}"
                f"\n\t\t\t\tPretrain epochs  : {args.ae_epochs}"
                f"\n\t\t\t\tAE-Learning rate : {args.ae_learning_rate}"
                f"\n\t\t\t\tAE-milestones    : {args.ae_lr_milestones}"
                f"\n\t\t\t\tAE-Weight decay  : {args.ae_weight_decay}\n"
                f"\n\t\t\t\tTrain epochs     : {args.epochs}"
                f"\n\t\t\t\tBatch size:      : {args.batch_size}"
                f"\n\t\t\t\tBatch acc.       : {args.batch_accumulation}"
                f"\n\t\t\t\tWarm up epochs   : {args.warm_up_n_epochs}"
                f"\n\t\t\t\tLearning rate    : {args.learning_rate}"
                f"\n\t\t\t\tMilestones       : {args.lr_milestones}"
                f"\n\t\t\t\tUse selectors    : {args.use_selectors}"
                f"\n\t\t\t\tWeight decay     : {args.weight_decay}\n"
                f"\n\t\t\t\tCode length      : {args.code_length}"
                f"\n\t\t\t\tNu               : {args.nu}"
                f"\n\t\t\t\tEncoder list     : {args.idx_list_enc}\n"
                f"\n\t\t\t\tTest metric      : {args.metric}"
            )
    else:
        if args.model_ckp is None:
            logger.info("CANNOT TEST MODEL WITHOUT A VALID CHECKPOINT")
            sys.exit(0)
        if args.debug:
            args.normal_class = 'carpet'
        else:
            if os.path.isfile(args.model_ckp):
                args.normal_class = args.model_ckp.split('/')[-2].split('-')[2].split('_')[-1]
            else:
                args.normal_class = args.model_ckp.split('/')[-3]

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

    # Print data infos  
    only_test = args.test and not args.train and not args.pretrain
    logger.info("Dataset info:")
    logger.info(
            f"Dataset             : {args.dataset_name}"
            f"\n\t\t\t\tNormal class  : {args.normal_class}"
            f"\n\t\t\t\tBatch size    : {args.batch_size}"    
        )
    if not only_test:
        logger.info(
                f"TRAIN:"
                f"\n\t\t\t\tNumber of images  : {len(train_loader.dataset)}"
                f"\n\t\t\t\tNumber of batches : {len(train_loader.dataset)//args.batch_size}"
            )
    logger.info(
            f"TEST:"
            f"\n\t\t\t\tNumber of images  : {len(test_loader.dataset)}"
        )

    is_texture = args.normal_class in tuple(["carpet", "grid", "leather", "tile", "wood"])
    input_shape = (3, 64, 64) if is_texture else (3, 128, 128)
        
    ### PRETRAIN the full AutoEncoder
    ae_net_cehckpoint = None
    if args.pretrain:        
        out_dir, tmp = get_out_dir(args, pretrain=True, aelr=None, net_name='mvtec')
        tb_writer = SummaryWriter(os.path.join(args.output_path, args.dataset_name, str(args.normal_class), 'svdd/tb_runs_pretrain', tmp))
        
        # Init AutoEncoder
        ae_net = MVTecNet_AutoEncoder(input_shape=input_shape, code_length=args.code_length, use_selectors=args.use_selectors)
        
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
        out_dir, tmp = get_out_dir(args, pretrain=False, aelr=aelr, net_name='mvtec')
        
        tb_writer = SummaryWriter(os.path.join(args.output_path, args.dataset_name, str(args.normal_class), 'tb_runs_train', tmp))
        
        # Init the Encoder network
        encoder_net = load_mvtec_model_from_checkpoint(
                                                input_shape=input_shape, 
                                                code_length=args.code_length, 
                                                idx_list_enc=args.idx_list_enc, 
                                                use_selectors=args.use_selectors, 
                                                net_cehckpoint=ae_net_cehckpoint,
                                                purge_ae_params=True
                                            )
                            
        ## Eval/Load hyperspeheres centers
        centers = eval_spheres_centers(train_loader=train_loader, encoder_net=encoder_net, ae_net_cehckpoint=ae_net_cehckpoint, debug=args.debug)
        
        # If we do not select any layer, then use only the last one
        # Remove all hyperspheres' center but the last one
        if len(args.idx_list_enc) == 0: args.idx_list_enc = [-1]
        keys = np.asarray(list(centers.keys()))[args.idx_list_enc]
        centers_ = {k: v for k, v in centers.items() if k in keys}
        
        # Start training
        net_cehckpoint = train(net, train_dset, out_dir, tb_writer, device, centers_, is_texture, args)
        tb_writer.close()

    ### TEST the Encoder
    if args.test:
        if net_cehckpoint is None:
            net_cehckpoint = args.model_ckp

        # Init table to print resutls on shell
        table_s = PrettyTable()
        table_s.field_names = ['Path', 'Code length', 'Enc layer list', 'weight decay', 'Object class', 'Boundary', 'batch size', 'nu', 'AUC', 'Balanced acc']
        table_s.float_format = '0.3'
        table_h = PrettyTable()
        table_h.field_names = ['Path', 'Code length', 'Enc layer list', 'weight decay', 'Object class', 'Boundary', 'batch size', 'nu', 'AUC', 'Balanced acc']
        table_h.float_format = '0.3'

        # Init dataframe to store results
        out_df = pd.DataFrame()

        is_file = os.path.isfile(net_cehckpoint)
        if is_file:
            out_df = test_models(
                            test_loader=test_loader, 
                            net_cehckpoint=net_cehckpoint, 
                            tables=(table_s, table_h), 
                            out_df=out_df, 
                            is_texture=is_texture,
                            input_shape=input_shape, 
                            idx_list_enc=args.idx_list_enc, 
                            boundary=args.boundary, 
                            normal_class=args.normal_class, 
                            use_selectors=args.use_selectors, 
                            device=device, 
                            debug=args.debug
                        )
        else:
            for model_ckp in tqdm(os.listdir(net_cehckpoint), total=len(os.listdir(net_cehckpoint)), desc="Running on models"):
                out_df = test_models(
                                test_loader=test_loader, 
                                net_cehckpoint=os.path.join(net_cehckpoint, model_ckp, 'best_oc_model_model.pth'), 
                                tables=(table_s, table_h), 
                                out_df=out_df, 
                                is_texture=is_texture, 
                                idx_list_enc=args.idx_list_enc, 
                                boundary=args.boundary, 
                                normal_class=args.normal_class, 
                                use_selectors=args.use_selectors, 
                                device=device, 
                                debug=args.debug
                            )

        print(table_s)
        print(table_h)

        normal_class = net_cehckpoint.split('/')[-3]
        b_path = "./mvtec_test_results/test_csv"
        ff = glob.glob(os.path.join(b_path, f'*{normal_class}*'))
        if len(ff) == 0:
            csv_out_name = os.path.join(b_path, f"test-results-{normal_class}_0.csv")
        else:
            ff.sort()
            version = int(ff[-1].split('_')[-1].split('.')[0]) + 1
            logger.info(f"Already found csv file for {normal_class} with latest version: {version-1} ==> creaing new csv file with version: {version}")
            csv_out_name = os.path.join(b_path, f"test-results-{normal_class}_{version}.csv")
        out_df.to_csv(csv_out_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AD')
    ## General config
    parser.add_argument('-s', '--seed', type=int, default=-1, help='Random seed (default: -1)')
    parser.add_argument('--n_workers', type=int, default=8, help='Number of workers for data loading. 0 means that the data will be loaded in the main process. (default: 8)')
    parser.add_argument('--output_path', default='./output/mvtec_ad')
    parser.add_argument('-lf', '--log-frequency', type=int, default=5, help='Log frequency (default: 5)')
    parser.add_argument('-dl', '--disable-logging', action="store_true", help='Disabel logging (default: False)')
    ## Model config
    parser.add_argument('-zl', '--code-length', default=64, type=int, help='Code length (default: 64)')
    parser.add_argument('-ck', '--model-ckp', help='Model checkpoint')
    ## Optimizer config
    parser.add_argument('-alr', '--ae-learning-rate', type=float, default=1.e-4, help='Warm up learning rate (default: 1.e-4)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1.e-4, help='Learning rate (default: 1.e-4)')
    parser.add_argument('-awd', '--ae-weight-decay', type=float, default=0.5e-6, help='Warm up learning rate (default: 0.5e-4)')
    parser.add_argument('-wd', '--weight-decay', type=float, default=0.5e-6, help='Learning rate (default: 0.5e-6)')
    parser.add_argument('-aml', '--ae-lr-milestones', type=int, nargs='+', default=[], help='Pretrain milestone')
    parser.add_argument('-ml', '--lr-milestones', type=int, nargs='+', default=[], help='Training milestone')
    ## Training config
    parser.add_argument('-we', '--warm_up_n_epochs', type=int, default=5, help='Warm up epochs (default: 5)')
    parser.add_argument('--use-selectors', action="store_true", help='Use features selector (default: False)')
    parser.add_argument('-ba', '--batch-accumulation', type=int, default=-1, help='Batch accumulation (default: -1, i.e., None)')
    parser.add_argument('-ptr', '--pretrain', action="store_true", help='Pretrain model (default: False)')
    parser.add_argument('-tr', '--train', action="store_true", help='Train model (default: False)')
    parser.add_argument('-tt', '--test', action="store_true", help='Test model (default: False)')
    parser.add_argument('-dn', '--dataset-name', default='MVTec_Anomaly')
    parser.add_argument('-ul', '--unlabelled-data', action="store_true", help='Use unlabelled data (default: False)')
    parser.add_argument('-nc', '--normal-class', choices=('bottle', 'capsule', 'grid', 'leather', 'metal_nut', 'screw', 'toothbrush', 'wood', 'cable', 'carpet', 'hazelnut', 'pill', 'tile', 'transistor', 'zipper'), default='cable', help='Category (default: cable)')
    parser.add_argument('-tbc', '--train-best-conf', action="store_true", help='Train best configurations (default: False)')
    parser.add_argument('-db', '--debug', action="store_true", help='Debug (default: False)')
    parser.add_argument('-bs', '--batch-size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('-bd', '--boundary', choices=("hard", "soft"), default="soft", help='Boundary (default: soft)')
    parser.add_argument('-ile', '--idx-list-enc', type=int, nargs='+', default=[], help='List of indices of model encoder')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Training epochs (default: 1)')
    parser.add_argument('-ae', '--ae-epochs', type=int, default=1, help='Warmp up epochs (default: 1)')
    parser.add_argument('-nu', '--nu', type=float, default=0.1)
    ## Test config
    parser.add_argument('-mt', '--metric', choices=(1, 2), type=int, default=2, help="Metric to evaluate norms (default: 2, i.e., L2)")
    args = parser.parse_args()

    main(args)
