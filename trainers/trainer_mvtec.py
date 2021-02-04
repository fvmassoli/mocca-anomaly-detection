import os
import sys
import time
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from sklearn.metrics import roc_curve, roc_auc_score, auc


def pretrain(ae_net: nn.Module, train_loader: DataLoader, out_dir: str, tb_writer: SummaryWriter, device: str, ae_learning_rate: float, ae_weight_decay: float, ae_lr_milestones: list, ae_epochs: int, log_frequency: int, batch_accumulation: int):
    """Train the full AutoEncoder network.

    Parameters
    ----------
    ae_net : nn.Module
        AutoEncoder network
    train_loader : DataLoader
        Data laoder
    out_dir : str
        Path to checkpoint dir
    tb_writer : SummaryWriter
        Writer on tensorboard
    device : str 
        Device
    ae_learning_rate : float
        AutoEncoder learning rate
    ae_weight_decay : float
        Weight decay
    ae_lr_milestones : list
        Epochs at which drop the learning rate
    ae_epochs: int
        Number of training epochs
    log_frequency : int
        Number of iteration after which show logs
    batch_accumulation : int
        Number of iteration among which accumulate gradients

    Returns
    -------
    ae_net_cehckpoint : str
        Path to model checkpoint

    """
    logger = logging.getLogger()
    
    ae_net = ae_net.train().to(device)

    optimizer = Adam(ae_net.parameters(), lr=ae_learning_rate, weight_decay=ae_weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=ae_lr_milestones, gamma=0.1)

    # Independent index to save stats on tensorboard
    kk = 1
    
    # Counter for the batch accumulation steps
    j_ba_steps = 0

    for epoch in range(ae_epochs):

        loss_epoch = 0.0
        n_batches = 0
        optimizer.zero_grad()
        for idx, (data, _) in enumerate(tqdm(train_loader, total=len(train_loader), leave=False)):
            if isinstance(data, list): data = data[0]

            data = data.to(device)
            x_r = ae_net(data)
            scores = torch.sum((x_r - data) ** 2, dim=tuple(range(1, x_r.dim())))
            loss = torch.mean(scores)
            loss.backward()

            j_ba_steps += 1
            if batch_accumulation != -1:
                if j_ba_steps % batch_accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    j_ba_steps = 0
            else:
                optimizer.step()
                optimizer.zero_grad()

            # Sanity check
            if np.isnan(loss.item()):
                logger.info("Found nan values into loss")
                sys.exit(0)

            loss_epoch += loss.item()
            n_batches += 1

            if idx != 0 and idx % ((len(train_loader)//log_frequency)+1) == 0:
                logger.info(f"PreTrain at epoch: {epoch+1} ([{idx}]/[{len(train_loader)}]) ==> Recon Loss: {loss_epoch/idx:.4f}")
                tb_writer.add_scalar('pretrain/recon_loss', loss_epoch/idx, kk)
                kk += 1

        scheduler.step()
        if epoch in ae_lr_milestones:
            logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

        ae_net_cehckpoint = os.path.join(out_dir, f'best_ae_ckp.pth')
        torch.save({'ae_state_dict': ae_net.state_dict()}, ae_net_cehckpoint)
        logger.info(f'Saved best autoencoder so far at: {ae_net_cehckpoint}')

    logger.info('Finished pretraining.')

    return ae_net_cehckpoint


def train(net: torch.nn.Module, train_loader: DataLoader, out_dir: str, tb_writer: SummaryWriter, device: str, ae_net_cehckpoint: str, idx_list_enc: list, learning_rate: float, weight_decay: float, lr_milestones: list, epochs: int, nu: float, boundary: str):
    """Train the Encoder network on the one class task.

    Parameters
    ----------
    net : nn.Module
        Encoder network
    train_loader : DataLoader
        Data laoder
    out_dir : str
        Path to checkpoint dir
    tb_writer : SummaryWriter
        Writer on tensorboard
    device : str 
        Device
    ae_net_cehckpoint : str 
        Path to autoencoder checkpoint
    idx_list_enc : list
        List of indexes of layers from which extract features
    learning_rate : float
        AutoEncoder learning rate
    weight_decay : float
        Weight decay
    lr_milestones : list
        Epochs at which drop the learning rate
    epochs: int
        Number of training epochs
    nu : float
        Value of the trade-off parameter
    boundary : str
        Type of boundary

    Returns
    -------
    net_cehckpoint : str
        Path to model checkpoint

    """
    logger = logging.getLogger()
    
    optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

    # Init spheres' radius
    R = {k: torch.tensor(0.0, device=device) for k in centers.keys()}

    # Training
    logger.info('Start training...')
    kk = 1
    net.train().to(device)
    best_loss = 1.e6
    for epoch in range(epochs):

        j = 0
        loss_epoch = 0.0
        n_batches = 0
        d_from_c = {}
        optimizer.zero_grad()
        for idx, (data, _) in enumerate(tqdm(train_loader, total=len(train_loader), leave=False)):
            if isinstance(data, list): data = data[0]

            data = data.to(device)

            # Update network parameters via backpropagation: forward + backward + optimize
            zipped = net(data)
            
            dist, loss = eval_ad_loss(zipped, centers, R, args.nu, args.boundary)

            for k in dist.keys():
                if k not in d_from_c:
                    d_from_c[k] = 0
                d_from_c[k] += torch.mean(dist[k]).item()

            loss.backward()
            j += 1
            if args.batch_accumulation != -1:
                if j == args.batch_accumulation:
                    j = 0
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()

            # Update hypersphere radius R on mini-batch distances
            if (args.boundary == 'soft') and (epoch >= warm_up_n_epochs):
                # R.data = torch.tensor(get_radius(dist, args.nu), device=device)
                for k in R.keys():
                    R[k].data = torch.tensor(
                                        np.quantile(np.sqrt(dist[k].clone().data.cpu().numpy()), 1 - args.nu), 
                                        device=device
                                    )

            loss_epoch += loss.item()
            n_batches += 1

            if args.debug and idx == 2:
                break
            if np.isnan(loss.item()):
                logger.info("Found nan values into loss")
                sys.exit(0)

            if idx != 0 and idx % ((len(train_loader)//args.log_frequency)+1) == 0:
                # log epoch statistics
                logger.info(f"TRAIN at epoch: {epoch+1} ([{idx}]/[{len(train_loader)}]) ==> Objective Loss: {loss_epoch/idx:.4f}")
                tb_writer.add_scalar('train/objective_loss', loss_epoch/idx, kk)
                for en, k in enumerate(d_from_c.keys()):
                    logger.info(
                        f"[{k}] -- Radius: {R[k]:.4f} - "
                        f"Dist from sphere centr: {d_from_c[k]/idx:.4f}"
                    )
                    tb_writer.add_scalar(f'train/radius_{k}', R[k], kk)
                    tb_writer.add_scalar(f'train/distance_c_sphere_{k}', d_from_c[k]/idx, kk)
                kk += 1

        scheduler.step()
        if epoch in args.lr_milestones:
            logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

        if (loss_epoch/len(train_loader)) <= best_loss:
            net_cehckpoint = os.path.join(out_dir, f'best_oc_model_model.pth')
            best_loss = (loss_epoch/len(train_loader))
            torch.save({
                    'net_state_dict': net.state_dict(),
                    'R': R,
                    'c': centers
                }, 
                net_cehckpoint
            )
            logger.info(f'Saved best model so far at: {net_cehckpoint}')

    logger.info('Finished training!!')
    
    return net_cehckpoint
        
def test(category: str, is_texture: bool, net: torch.nn.Module, test_loader: DataLoader, R: dict, c: dict, device: str, idx_list_enc: list, boundary: str):
    """Test the Encoder network.

    Parameters
    ----------
    category : str
        Name of the class under test
    is_texture : bool
        True if the input data belong to a texture-type class
    net : nn.Module
        Encoder network
    test_loader : DataLoader
        Data laoder
    R : dict
        Dictionary containing the values of the radiuses for each layer
    c : dict
        Dictionary containing the values of the hyperspheres' center for each layer
    device : str 
        Device
    idx_list_enc : list
        List of indexes of layers from which extract features
    boundary : str
        Type of boundary

    Returns
    -------
    test_auc : float
        AUC
    balanced_accuracy : float
        Maximum Balanced Accuracy

    """
    logger = logging.getLogger()

    # Testing
    logger.info('Start testing...')
    
    idx_label_score = []
    net.eval().to(device)
    with torch.no_grad():
        for idx, (data, labels) in enumerate(tqdm(test_loader, total=len(test_loader), desc=f"Testing class: {category}", leave=False)):
            data = data.to(device)
            
            if texture:
                ## Get 8 patches from each texture image ==> the anomaly score is max{score(pathes)}
                _, _, h, w = data.shape
                assert h == w, "Height and Width are different!!!"
                patch_size = 64
                nb_patches = (h // patch_size) ** 2
                patches = []
                start_w = 0
                start_h = 0
                for _ in range(nb_patches):
                    patches.append(data[:, :, start_h:start_h+patch_size, start_w:start_w+patch_size])
                    start_w += patch_size
                    if start_w == w:
                        start_h += patch_size
                        start_w = 0
                patches = torch.stack(patches, dim=1) 
                # patches.shape = (b_size, nb_patches, ch, h, w)
                # batch.shape = (nb_patches, ch, h, w)
                scores = torch.stack([get_scores(net(batch), c, R, device, boundary, args.metric, textures=True) for batch in patches])
            else:
                zipped = net(data)
                scores = get_scores(zipped, c, R, device, boundary, args.metric, textures=False)

            idx_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist()))

            if args.debug and idx == 2:
                break

    # Compute AUC
    labels, scores = zip(*idx_label_score)
    labels = np.array(labels)
    scores = np.array(scores)

    #test_auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    balanced_accuracy = np.max((tpr + (1 - fpr)) / 2)
    auroc = auc(fpr, tpr)
    balanced_accuracy = np.max((tpr + (1 - fpr)) / 2)
    logger.info('Test set AUC - B: {:.4f} - {:.4f}'.format(auroc, balanced_accuracy))
    logger.info('Finished testing!!')

    return auroc, balanced_accuracy


def eval_ad_loss(zipped: dict, c: dict, R: dict, nu: float, boundary: str):
    """Evaluate ancoder loss in the one class setting. 
    
    Parameters
    ----------
    zipped : dict
    
    c : dict
    
    R : dict 
    
    nu : float
    
    boundary: str


    Returns
    -------


    """"
    dist = {}
    loss = 1
    if isinstance(zipped, torch.Tensor):
        zipped = [('06', zipped)] # it is the code z of the encoder
    for item in zipped:
        k = item[0]
        v = item[1]
        dist[k] = torch.sum((v - c[k].unsqueeze(0)) ** 2, dim=1)
        if boundary == 'soft':
            scores = dist[k] - R[k] ** 2
            loss += R[k] ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            loss += torch.mean(dist[k])
    return dist, loss


def get_scores(zipped: dict, c: dict, R: dict, device: str, boundary: str, metric: str, is_textures: bool):
    """Evaluate anomaly score. 
    
    Parameters
    ----------
    zipped : dict
    
    c : dict
    
    R : dict
    
    device : str
    
    boundary : str
    
    metric : str
    
    is_textures: bool


    Returns
    -------


    """
    zipped = [('06', zipped)] if isinstance(zipped, torch.Tensor) else zipped
    dist = {item[0]: torch.norm(item[1] - c[item[0]].unsqueeze(0), p=metric, dim=1)**metric for item in zipped}
    shape = dist[list(dist.keys())[0]].shape[0]
    scores = torch.zeros((shape,), device=device)
    for k in dist.keys():
        if boundary == 'soft':
            scores += dist[k] - R[k] ** metric # R[k] is a number not a vector
        else:
            scores += dist[k]
    return scores.max()/len(list(dist.keys())) if textures else scores/len(list(dist.keys()))
