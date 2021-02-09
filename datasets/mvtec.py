import os
import sys
import math
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from os.path import join

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torchvision.transforms as T
from torchvision.datasets import ImageFolder


class MVtecDataset(ImageFolder):
    """Torchvision ImageFolder class with patch of __getitem__ method to targets according to the task.

    """
    def __init__(self, root: str, transform):
        super(MVtecDataset, self).__init__(root=root, transform=transform)

        # Index of the class that corresponds to the folder named 'good'
        self.normal_class_idx = self.class_to_idx['good']

    def __getitem__(self, index: int):
        data, target = self.samples[index]
        def read_image(path):
            """Returns the image in RGB
            
            """
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')

        # Convert the target to the 0/1 case
        target = 0 if target == self.normal_class_idx else 1
        data = self.transform(read_image(data))

        return data, target
        

class CustomTensorDataset(TensorDataset):
    """Custom dataset for preprocessed images.
    
    """
    def __init__(self, root: str):
        """Init the dataset.

        Parameters
        ----------
        root : str
            Path to data file
        
        """
        # Load data
        self.data = torch.from_numpy(np.load(root))
        
        # Load TensorDataset
        super(CustomTensorDataset, self).__init__(self.data)
        
    def __len__(self):
        return self.tensors[0].size(0)
    
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors), 0
  

class MVTec_DataHolder(object):
    """MVTec data holder class
    
    """
    def __init__(self, data_path: str, category: str, image_size: int, patch_size: int, rotation_range: tuple, is_texture: bool):
        """Init MVTec data holder class
        
        Parameters
        ----------
        category : str
            Normal class 
        image_size : int
            Side size of the input images
        patch_size : int
            Side size of the patches (for textures only)
        rotation_range : tuple
            Min and max angle to rotate images
        is_texture : bool
            True if the category is texture-type class

        """"
        self.data_path = data_path
        self.category = normal_class 
        self.image_size = image_size 
        self.patch_size = patch_size 
        self.rotation_range = rotation_range 
        self.is_texture = is_texture

    def get_test_data(self):
        """Load test dataset

        Returns
        -------
        MVtecDataset : Dataset
            Custom dataset to handle MVTec data

        """
        return MVtecDataset(
                        root=join(self.data_path, f'MVTec_Anomaly/{category}/test'),
                        transform=T.Compose([
                                        T.Resize(image_size, interpolation=Image.BILINEAR),
                                        T.ToTensor(),
                                        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                    ])
                    )
        
    def get_train_data(self, return_dataset: bool=True):
        """Load train dataset

        Parameters
        ----------
        return_dataset : bool
            False for preprocessing purpose only

        """
        train_data_dir = join(self.data_path, f'MVTec_Anomaly/{category}/train/')
        
        # Preprocessed output data path
        cache_main_dir = join(self.data_path, f'MVTec_Anomaly/processed/{category}')
        os.makedirs(cache_main_dir, exist_ok=True)
        cache_file = f'{cache_main_dir}/{self.category}_train_dataset_i-{self.image_size}_p-{self.patch_size}_r-{self.rotation_range[0]}--{self.rotation_range[1]}.npy'

        # Check if preprocessed file already exists
        if not os.path.exists(cache_file):

            # Apply random rotation
            def augmentation(): 
                """Returns transforms to apply to the data
                
                """
                # For textures rotate and crop without edges
                if self.is_texture:
                    return T.Compose([
                                T.Resize(self.image_size, interpolation=Image.BILINEAR),
                                T.Pad(padding=self.image_size//4, padding_mode="reflect"),
                                T.RandomRotation((self.rotation_range[0], self.rotation_range[1])),
                                T.CenterCrop(self.image_size),
                                T.RandomCrop(self.patch_size),
                                T.ToTensor(),
                                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                            ])
                else:
                    return T.Compose([
                                T.Resize(self.image_size, interpolation=Image.BILINEAR),
                                T.Pad(padding=self.image_size//4, padding_mode="reflect"),
                                T.RandomRotation((self.rotation_range[0], self.rotation_range[1])),
                                T.CenterCrop(self.image_size),
                                T.ToTensor(),
                                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                            ])
                
            # Load data and apply transformations
            train_dataset = ImageFolder(root=train_data_dir, transform=augmentation())
            print(f"Creating cache for dataset: \n{cache_file}")
            # To simulate a larger datasegt, replicate images with transformations
            nb_epochs = 50000 // len(train_dataset.imgs)
            data_loader = DataLoader(dataset=train_dataset, batch_size=1024, pin_memory=True)

            for epoch in tqdm(range(nb_epochs), total=nb_epochs, desc=f"Creating cache for: {category}"):
                if epoch == 0:
                    cache_np = [x.numpy() for x, _ in tqdm(data_loader, total=len(data_loader), desc=f'Caching epoch: {epoch+1}/{nb_epochs+1}', leave=False)]
                else:
                    cache_np.extend([x.numpy() for x, _ in tqdm(data_loader, total=len(data_loader), desc=f'Caching epoch: {epoch+1}/{nb_epochs+1}', leave=False)])
            
            cache_np = np.vstack(cache_np)
            np.save(cache_file, cache_np)
            print(f"Preprocessed images has been saved at: \n{cache_file}")
        
        if return_dataset:
            print(f"Loading dataset from cache: \n{cache_file}")
            return CustomTensorDataset(cache_file)
        else:
            return

    def get_loaders(self, batch_size: int, shuffle_train: bool=True, pin_memory: bool=False, num_workers: int = 0):
        """Returns MVtec dataloaders
        
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
        
        Returns
        -------
        loaders : DataLoader
            Train and test data loaders

        """
        train_loader = DataLoader(
                            dataset=self.get_train_data(return_dataset=True), 
                            batch_size=batch_size, 
                            shuffle=shuffle_train,
                            pin_memory=pin_memory,
                            num_workers=num_workers
                        )
        test_loader = DataLoader(
                            dataset=self.get_test_data(), 
                            batch_size=batch_size,
                            pin_memory=pin_memory,
                            num_workers=num_workers
                        )
        return train_loader, test_loader


if __name__ == '__main__':
    """To speed up the train phase we can preprocess the training images and save them as numpy array.
    
    """
    textures = tuple(['carpet', 'grid', 'leather', 'tile', 'wood'])
    objects_1 = tuple(['bottle', 'hazelnut', 'metal_nut', 'screw'])
    objects_2 = tuple(['capsule', 'toothbrush', 'cable', 'pill', 'transistor', 'zipper'])
    
    classes = list(textures)
    classes.extends(list(objects_1))
    classes.extends(list(objects_2)) 

    for category in classes:
        if category in textures:
            args = dict(
                    category=category, 
                    image_size=512, 
                    patch_size=64, 
                    rotation_range=(0, 45), 
                    texture=True
                )
        elif category in objects_1:
            args = dict(
                    category=category, 
                    image_size=128, 
                    patch_size=-1, 
                    rotation_range=(-45, 45), 
                    texture=True
                )
        else:
            args = dict(
                    category=category, 
                    image_size=128, 
                    patch_size=-1, 
                    rotation_range=(0, 0), 
                    texture=False
                )

    MVTec_DataHolder(*args).get_train_data(return_dataset=False)
