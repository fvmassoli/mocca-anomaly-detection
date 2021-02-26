import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Subset

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


def get_target_label_idx(labels: np.array, targets: np.array):
    """Get the indices of labels that are included in targets.
    
    Parameters
    ----------
    labels : np.array
        Array of labels
    targets : np.array
        Array of target labels
    
    Returns
    ------
    List with indices of target labels
    
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


def global_contrast_normalization(x: torch.tensor, scale: str='l1') -> torch.Tensor:
    """Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).

    Parameters
    ----------
    x : torch.tensor
        Data sample
    scale : str
        Scale

    Returns
    ------
    Normalized features

    """
    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    # Evaluate the mean over all features (pixels) per sample
    mean = torch.mean(x)  
    x -= mean

    x_scale = torch.mean(torch.abs(x)) if scale == 'l1' else torch.sqrt(torch.sum(x ** 2)) / n_features

    return x / x_scale


class CIFAR10_DataHolder(object):
    """CIFAR10 data holder class
    
    """
    def __init__(self, root: str, normal_class=5):
        """Init CIFAR10 data holder class
        
        Parameters
        ----------
        root : str
            Path to root folder of the data
        normal_class : 
            Index of the normal class

        """
        self.root = root

        # Total number of classes = 2, i.e., 0: normal, 1: anomalies 
        self.n_classes = 2 
        
        # Tuple containing the normal classes
        self.normal_classes = tuple([normal_class])
        
        # List of the anomalous classes
        self.anomaly_classes = list(range(0, 10))
        self.anomaly_classes.remove(normal_class)

        # Init the datasets
        self.__init_train_test_datasets(normal_class)

    def __init_train_test_datasets(self, normal_class: int) -> None:
        """Init the datasets.
        
        Parameters
        ----------
        normal_class : int
            The index of the non-anomalous class
        
        """
        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [
                (-28.94083453598571, 13.802961825439636),
                (-6.681770233365245, 9.158067708230273),
                (-34.924463588638204, 14.419298165027628),
                (-10.599172931391799, 11.093187820377565),
                (-11.945022995801637, 10.628045447867583),
                (-9.691969487694928, 8.948326776180823),
                (-9.174940012342555, 13.847014686472365),
                (-6.876682005899029, 12.282371383343161),
                (-15.603507135507172, 15.2464923804279),
                (-6.132882973622672, 8.046098172351265)
        ]

        # Define CIFAR-10 preprocessing operations
        #   1. GCN with L1 norm 
        #   2. min-max feature scaling to [0,1]
        self.transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize(
                                                            [min_max[normal_class][0]] * 3,
                                                            [min_max[normal_class][1] - min_max[normal_class][0]] * 3
                                                        )
                                    ])

        # Define CIFAR-10 preprocessing operations on the labels,
        # i.e., set to 0 all the labels that belong to the anomalous classes
        self.target_transform = transforms.Lambda(lambda x: int(x in self.anomaly_classes))

        # Init training set
        self.train_set = MyCIFAR10(
                                root=self.root, 
                                train=True, 
                                download=True,
                                transform=self.transform, 
                                target_transform=self.target_transform
                            )

        # Subset the training set by considering normal class images only
        train_idx_normal = get_target_label_idx(labels=self.train_set.targets, targets=self.normal_classes)
        self.train_set = Subset(self.train_set, train_idx_normal)

        # Init test set
        self.test_set = MyCIFAR10(
                                root=self.root, 
                                train=False, 
                                download=True,
                                transform=self.transform, 
                                target_transform=self.target_transform
                            )

    def get_loaders(self, batch_size: int, shuffle_train: bool=True, pin_memory: bool=False, num_workers: int = 0) -> [torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Returns CIFAR10 dataloaders

        Parameters
        ----------
        batch_size : int
            Size of the batch to
        shuffle_train : bool
            If True, shuffles the training dataset
        pin_memory : bool
            If True, pin memeory
        num_workers : int 
            Number of dataloader workers
        
        Retunrs
        -------
        loaders : DataLoader
            Train and test data loaders

        """
        train_loader = DataLoader(
                            dataset=self.train_set, 
                            batch_size=batch_size, 
                            shuffle=shuffle_train,
                            pin_memory=pin_memory,
                            num_workers=num_workers
                        )
        test_loader = DataLoader(
                            dataset=self.test_set, 
                            batch_size=batch_size,
                            pin_memory=pin_memory,
                            num_workers=num_workers
                        )
        return train_loader, test_loader


class MyCIFAR10(CIFAR10):
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample.
    
    """
    def __init__(self, *args, **kwargs):
        super(MyCIFAR10, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.

        Parameters
        ----------
        index : int
            Index

        Returns
        -------
            triple: (image, target, index) where target is the index of the target class.

        """
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, index
