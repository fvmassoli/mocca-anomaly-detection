from glob import glob
from tqdm import tqdm
from time import time
from typing import List, Tuple
from os.path import basename, isdir, join, splitext

import cv2
import numpy as np
import skimage.io as io

import torch
from torchvision import transforms
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from .shanghaitech_test import ShanghaiTechTestHandler

class ShanghaiTech_DataHolder(object):
    """
    ShanghaiTech data holder class

    Parameters
    ----------
    root : str
        root folder of ShanghaiTech dataset
    clip_length : int
        number of frames that form a clip
    stide : int
        for creating a clip what should be size of sliding window
    """
    def __init__(self, root: str, clip_length=16, stride=1):
        self.root = root
        self.clip_length = clip_length
        self.stride = stride 
        self.shape = (3, clip_length, 256, 512)
        self.train_dir = join(root, 'training', 'nobackground_frames_resized')
        # Transform 
        self.transform = transforms.Compose([ToFloatTensor3D(normalize=True)])
        

    def get_test_data(self) -> Dataset:
        """Load test dataset

        Returns
        -------
        ShanghaiTech : Dataset
            Custom dataset to handle ShanghaiTech data

        """
        return ShanghaiTechTestHandler(self.root)

    def get_train_data(self, return_dataset: bool=True):
        """Load train dataset

        Parameters
        ----------
        return_dataset : bool
            False for preprocessing purpose only
        """
        
        if return_dataset:
            # Load all ids
            self.train_ids = self.load_train_ids()
            # Create clips with given clip_length and stride
            self.train_clips = self.create_clips(self.train_dir, self.train_ids, clip_length=self.clip_length, stride=self.stride, read_target=False)
                    
            return MySHANGHAI(self.train_clips, self.transform, clip_length=self.clip_length) 
        else:
            return

    def get_loaders(self, batch_size: int, shuffle_train: bool=True, pin_memory: bool=False, num_workers: int = 0) -> [DataLoader, DataLoader]:
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

    def load_train_ids(self):
        # type: () -> List[str]
        """
        Loads the set of all train video ids.
        :return: The list of train ids.
        """
        return sorted([basename(d) for d in glob(join(self.train_dir, '**')) if isdir(d)])

    def create_clips(self, dir_path, ids, clip_length=16, stride=1, read_target=False):
        # type: (str, int, int, bool)
        """
        Gets frame directory and ids of the directories in the frame dir
        Creates clips which consist of number of clip_length at each clip.
        Clips are created in a sliding window fashion. Default window slide is 1
        but stride controls the window slide
        Example: for default parameters first clip is [001.jpg, 002.jpg, ...,016.jpg]
        second clip would be [002.jpg, 003.jpg, ..., 017.jpg]
        If read_target is True then it will try to read from test directory 
        If read_target is False then it will populate the array with all zeros
        :return: clips:: numpy array with (num_clips,clip_length) shape
                 ground_truths:: numpy array with (num_clips,clip_length) shape
        """ 
        clips = []
        print(f"Creating clips for {dir_path} dataset with length {clip_length}...")
        for idx in tqdm(ids):
            frames = sorted([x for x in glob(join(dir_path, idx, "*.jpg"))])
            num_frames = len(frames)
            # Slide the window with stride to collect clips
            for window in range(0, num_frames-clip_length+1, stride):
                clips.append(frames[window:window+clip_length])
        return np.array(clips)

class MySHANGHAI(Dataset):
    def __init__(self, clips, transform=None, clip_length=16):
        self.clips = clips
        self.transform = transform
        self.shape = (3, clip_length, 256, 512)

    def __len__(self):
        return 10000 # len(self.clips)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
            targets are all 0 target
        """
        index_ = torch.randint(0, len(self.clips), size=(1,)).item()
        sample = np.stack([np.uint8(io.imread(img_path)) for img_path in self.clips[index_]])
        sample = self.transform(sample) if self.transform else sample
        return sample, index_ 

from scipy.ndimage.morphology import binary_dilation


def get_target_label_idx(labels, targets):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
    Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features

    x /= x_scale

    return x

class ToFloatTensor3D(object):
    """ Convert videos to FloatTensors """
    def __init__(self, normalize=True):
        self._normalize = normalize

    def __call__(self, sample):
        if len(sample) == 3:
            X, Y, _ = sample
        else:
            X = sample

        # swap color axis because
        # numpy image: T x H x W x C
        X = X.transpose(3, 0, 1, 2)
        #Y = Y.transpose(3, 0, 1, 2)

        if self._normalize:
            X = X / 255.
         
        X = np.float32(X)
        return torch.from_numpy(X)

class ToFloatTensor3DMask(object):
    """ Convert videos to FloatTensors """
    def __init__(self, normalize=True, has_x_mask=True, has_y_mask=True):
        self._normalize = normalize
        self.has_x_mask = has_x_mask
        self.has_y_mask = has_y_mask

    def __call__(self, sample):
        X = sample
        # swap color axis because
        # numpy image: T x H x W x C
        X = X.transpose(3, 0, 1, 2)

        X = np.float32(X)

        if self._normalize:
            if self.has_x_mask:
                X[:-1] = X[:-1] / 255.
            else:
                X = X / 255.

        return torch.from_numpy(X)


class RemoveBackground:

    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, sample: tuple) -> tuple:
        X, Y, background = sample

        mask = np.uint8(np.sum(np.abs(np.int32(X) - background), axis=-1) > self.threshold)
        mask = np.expand_dims(mask, axis=-1)

        mask = np.stack([binary_dilation(mask_frame, iterations=5) for mask_frame in mask])

        X *= mask

        return X, Y, background