import os
import time
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader

from tensorboardX import SummaryWriter

from sklearn.metrics import roc_auc_score


def pretrain(ae_net: torch.nn.Module, train_loader: DataLoader, out_dir: str, tb_writer: SummaryWriter, device: str, ae_learning_rate: float, ae_weight_decay: float, ae_lr_milestones: list, ae_epochs: int) -> str:
    """Train the full AutoEncoder network.

    Parameters
    ----------
    ae_net : torch.nn.Module
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

    Returns
    -------
    ae_net_cehckpoint : str
        Path to model checkpoint

    """
    logger = logging.getLogger()
    
    ae_net = ae_net.train().to(device)

    optimizer = Adam(ae_net.parameters(), lr=ae_learning_rate, weight_decay=ae_weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=ae_lr_milestones, gamma=0.1)

    for epoch in range(ae_epochs):
        loss_epoch = 0.0
        n_batches = 0

        for (data, _, _) in train_loader:
            data = data.to(device)
            
            optimizer.zero_grad()

            outputs = ae_net(data)
            
            scores = torch.sum((outputs - data) ** 2, dim=tuple(range(1, outputs.dim())))
            
            loss = torch.mean(scores)
            loss.backward()
            
            optimizer.step()

            loss_epoch += loss.item()
            n_batches += 1

        scheduler.step()
        if epoch in ae_lr_milestones:
            logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

        logger.info(f"PreTrain at epoch: {epoch+1} ==> Recon Loss: {loss_epoch/len(train_loader):.4f}")
        tb_writer.add_scalar('pretrain/recon_loss', loss_epoch/len(train_loader), epoch+1)

    logger.info('Finished pretraining.')

    ae_net_cehckpoint = os.path.join(out_dir, f'ae_ckp_{time.time()}.pth')
    torch.save({'ae_state_dict': ae_net.state_dict()}, ae_net_cehckpoint)
    logger.info(f'Saved autoencoder at: {ae_net_cehckpoint}')

    return ae_net_cehckpoint


def train(net: torch.nn.Module, train_loader: DataLoader, out_dir: str, tb_writer: SummaryWriter, device: str, ae_net_cehckpoint: str, idx_list_enc: list, learning_rate: float, weight_decay: float, lr_milestones: list, epochs: int, nu: float, boundary: str, debug: bool) -> str:
    """Train the Encoder network on the one class task.

    Parameters
    ----------
    net : torch.nn.Module
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
    debug: bool
        If True, enable debug mode

    Returns
    -------
    net_cehckpoint : str
        Path to model checkpoint

    """
    logger = logging.getLogger()
    
    net.train().to(device)

    # Hook model's layers
    feat_d = {}
    hooks = hook_model(idx_list_enc=idx_list_enc, model=net, dataset_name="cifar10", feat_d=feat_d)

    optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

    # Initialize hypersphere center c
    logger.info('Initializing center c...')
    c = init_center_c(feat_d=feat_d, train_loader=train_loader, net=net, device=device)
    logger.info('Center c initialized.')

    R = {k: torch.tensor(0.0, device=device) for k in c.keys()}

    logger.info('Start training...')
    warm_up_n_epochs = 10
    
    for epoch in range(epochs):
        loss_epoch = 0.0
        n_batches = 0
        d_from_c = {}
        
        for (data, _, _) in train_loader:
            data = data.to(device)

            # Update network parameters via backpropagation: forward + backward + optimize
            _ = net(data)
            
            dist, loss = eval_ad_loss(feat_d=feat_d, c=c, R=R, nu=nu, boundary=boundary)

            for k in dist.keys():
                if k not in d_from_c:
                    d_from_c[k] = 0
                d_from_c[k] += torch.mean(dist[k]).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update hypersphere radius R on mini-batch distances
            # only after the warm up epochs
            if (boundary == 'soft') and (epoch >= warm_up_n_epochs):
                for k in R.keys():
                    R[k].data = torch.tensor(
                                        np.quantile(np.sqrt(dist[k].clone().data.cpu().numpy()), 1 - nu), 
                                        device=device
                                    )

            loss_epoch += loss.item()
            n_batches += 1

        scheduler.step()
        if epoch in lr_milestones:
            logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

        # log epoch statistics
        logger.info(f"TRAIN at epoch: {epoch} ==> Objective Loss: {loss_epoch/n_batches:.4f}")
        tb_writer.add_scalar('train/objective_loss', loss_epoch/n_batches, epoch)
        for en, k in enumerate(d_from_c.keys()):
            logger.info(
                f"[{k}] -- Radius: {R[k]:.4f} - "
                f"Dist from sphere centr: {d_from_c[k]/n_batches:.4f}"
            )
            tb_writer.add_scalar(f'train/radius_{idx_list_enc[en]}', R[k], epoch)
            tb_writer.add_scalar(f'train/distance_c_sphere_{idx_list_enc[en]}', d_from_c[k]/n_batches, epoch)
        
    logger.info('Finished training!!')
    
    [h.remove() for h in hooks]

    time_ = ae_net_cehckpoint.split('_')[-1].split('.p')[0]
    net_cehckpoint = os.path.join(out_dir, f'net_ckp_{time_}.pth')
    if debug:
        net_cehckpoint = './test_net_ckp.pth'
    torch.save({
            'net_state_dict': net.state_dict(),
            'R': R,
            'c': c
        }, 
        net_cehckpoint
    )
    logger.info(f'Saved model at: {net_cehckpoint}')

    return net_cehckpoint
        

def test(net: torch.nn.Module, test_loader: DataLoader, R: dict, c: dict, device: str, idx_list_enc: list, boundary: str) -> float:
    """Test the Encoder network.

    Parameters
    ----------
    net : torch.nn.Module
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
    debug: bool
        If True, enable debug mode

    Returns
    -------
    test_auc : float
        AUC

    """
    logger = logging.getLogger()

    # Hook model's layers
    feat_d = {}
    hooks = hook_model(idx_list_enc=idx_list_enc, model=net, dataset_name="cifar10", feat_d=feat_d)

    # Testing
    logger.info('Starti testing...')
    idx_label_score = []
    net.eval().to(device)
    with torch.no_grad():
        for data in test_loader:
            inputs, labels, idx = data
            inputs = inputs.to(device)
            
            _ = net(inputs)

            scores = get_scores(feat_d=feat_d, c=c, R=R, device=device, boundary=boundary)

            # Save triples of (idx, label, score) in a list
            idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                        labels.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist()))

    [h.remove() for h in hooks]

    # Compute AUC
    _, labels, scores = zip(*idx_label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    test_auc = roc_auc_score(labels, scores)
    logger.info('Test set AUC: {:.2f}%'.format(100. * test_auc))

    logger.info('Finished testing!!')

    return 100. * test_auc


def hook_model(idx_list_enc: list, model: torch.nn.Module, dataset_name: str, feat_d: dict) -> None:
    """Create hooks for model's layers.
    
    Parameters
    ----------
    idx_list_enc : list
        List of indexes of layers from which extract features
    model : torch.nn.Module
        Encoder network
    dataset_name : str
        Name of the dataset
    feat_d : dict
        Dictionary containing features

    Returns
    -------
        registered hooks

    """
    if dataset_name == 'mnist':

        blocks_ = [model.conv1, model.conv2, model.fc1]
    else:
        
        blocks_ = [model.conv1, model.conv2, model.conv3, model.fc1]
    
    if isinstance(idx_list_enc, list) and len(idx_list_enc) != 0:
        assert len(idx_list_enc) <= len(blocks_), f"Too many indices for decoder: {idx_list_enc} - for {len(blocks_)} blocks"
        blocks = [blocks_[idx] for idx in idx_list_enc]

    blocks_idx = dict(zip(blocks, map('{:02d}'.format, range(len(blocks)))))
        
    def hook_func(module, input, output):
        block_num = blocks_idx[module]
        extracted = output
        if extracted.ndimension() > 2:
            extracted = F.avg_pool2d(extracted, extracted.shape[-2:])
        feat_d[block_num] = extracted.squeeze()
    
    return [b.register_forward_hook(hook_func) for b in blocks_idx] 


@torch.no_grad()
def init_center_c(feat_d: dict, train_loader: DataLoader, net: torch.nn.Module, device: str, eps: float = 0.1) -> dict:
    """Initialize hyperspheres' center c as the mean from an initial forward pass on the data.
    
    Parameters
    ----------
    feat_d : dict
        Dictionary containing features
    train_loader : DataLoader
        Training data loader
    net : torch.nn.Module
        Encoder network
    device : str
        Device
    eps : float = 0.1
        If a center is too close to 0, set to +-eps
    Returns
    -------
    c : dict
        hyperspheres' center

    """
    n_samples = 0
    
    net.eval()

    for idx, (data, _, _) in enumerate(tqdm(train_loader, desc='init hyperspheres centeres', total=len(train_loader), leave=False)):
        data = data.to(device)
    
        outputs = net(data)
        n_samples += outputs.shape[0]
        
        if idx == 0:
            c = {k: torch.zeros_like(feat_d[k][-1], device=device) for k in feat_d.keys()}
        
        for k in feat_d.keys():
            c[k] += torch.sum(feat_d[k], dim=0)
    
    for k in c.keys():
        c[k] = c[k] / n_samples
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[k][(abs(c[k]) < eps) & (c[k] < 0)] = -eps
        c[k][(abs(c[k]) < eps) & (c[k] > 0)] = eps
    
    return c


def eval_ad_loss(feat_d: dict, c: dict, R: dict, nu: float, boundary: str) -> [dict, torch.Tensor]:
    """Eval training loss.
    
    Parameters
    ----------
    feat_d : dict
        Dictionary containing features
    c : dict
        Dictionary hyperspheres' center
    R : dict
        Dictionary hyperspheres' radius
    nu : float
        Value of the trade-off parameter
    boundary : str
        Type of boundary

    Returns
    -------
    dist : dict
        Dictionary containing the average distance of the features vectors from the hypersphere's center for each layer
    loss : torch.Tensor
        Loss value
    """
    dist = {}
    loss = 1

    for k in feat_d.keys():
    
        dist[k] = torch.sum((feat_d[k] - c[k].unsqueeze(0)) ** 2, dim=1)
        if boundary == 'soft':
            
            scores = dist[k] - R[k] ** 2
            loss += R[k] ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            
            loss += torch.mean(dist[k])
    
    return dist, loss


def get_scores(feat_d: dict, c: dict, R: dict, device: str, boundary: str) -> float:
    """Eval anomaly score.

    Parameters
    ----------
    feat_d : dict
        Dictionary containing features
    c : dict
        Dictionary hyperspheres' center
    R : dict
        Dictionary hyperspheres' radius
    device : str
        Device
    boundary : str
        Type of boundary

    Returns
    -------
    scores : float
        Anomaly score

    """
    dist, _ = eval_ad_loss(feat_d, c, R, 1, boundary)
    shape = dist[list(dist.keys())[0]].shape[0]
    scores = torch.zeros((shape,), device=device)
    
    for k in dist.keys():
        if boundary == 'soft':
            
            scores += dist[k] - R[k] ** 2
        else:
            
            scores += dist[k]
    
    return scores/len(list(dist.keys()))
