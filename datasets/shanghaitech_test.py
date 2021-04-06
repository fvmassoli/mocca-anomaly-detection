from glob import glob
from os.path import basename
from os.path import isdir
from os.path import join
from typing import List
from typing import Tuple

import cv2
import numpy as np
import skimage.io as io

import torch

from skimage.transform import resize
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from .base import VideoAnomalyDetectionDataset, ToFloatTensor3D, ToFloatTensor3DMask 

class SHANGHAITECHTEST(VideoAnomalyDetectionDataset):
    """
    Models ShanghaiTech dataset for video anomaly detection.
    """
    def __init__(self, path):
        # type: (str) -> None
        """
        Class constructor.
        :param path: The folder in which ShanghaiTech is stored.
        """
        super(SHANGHAITECHTEST, self).__init__()
        self.path = path
        # Test directory
        self.test_dir = join(path, 'shanghaitech', 'testing')
        # Transform
        self.transform = transforms.Compose([ToFloatTensor3D(normalize=True)])
        # Load all test ids
        self.test_ids = self.load_test_ids()
        # Other utilities
        self.cur_len = 0
        self.cur_video_id = None
        self.cur_video_frames = None
        self.cur_video_gt = None

    def load_test_ids(self):
        # type: () -> List[str]
        """
        Loads the set of all test video ids.
        :return: The list of test ids.
        """
        return sorted([basename(d) for d in glob(join(self.test_dir, 'nobackground_frames_resized', '**')) if isdir(d)])

    def load_test_sequence_frames(self, video_id):
        # type: (str) -> np.ndarray
        """
        Loads a test video in memory.
        :param video_id: the id of the test video to be loaded
        :return: the video in a np.ndarray, with shape (n_frames, h, w, c).
        """
        c, t, h, w = self.shape
        sequence_dir = join(self.test_dir,  'nobackground_frames_resized', video_id)
        img_list = sorted(glob(join(sequence_dir, '*.jpg')))
        #print(f"Creating clips for {sequence_dir} dataset with length {t}...")
        return np.stack([np.uint8(io.imread(img_path)) for img_path in img_list])

    def load_test_sequence_gt(self, video_id):
        # type: (str) -> np.ndarray
        """
        Loads the groundtruth of a test video in memory.
        :param video_id: the id of the test video for which the groundtruth has to be loaded.
        :return: the groundtruth of the video in a np.ndarray, with shape (n_frames,).
        """
        clip_gt = np.load(join(self.test_dir,  'test_frame_mask', f'{video_id}.npy'))
        return clip_gt

    def test(self, video_id):
        # type: (str) -> None
        """
        Sets the dataset in test mode.
        :param video_id: the id of the video to test.
        """
        c, t, h, w = self.shape
        self.cur_video_id = video_id
        self.cur_video_frames = self.load_test_sequence_frames(video_id)
        self.cur_video_gt = self.load_test_sequence_gt(video_id)
        self.cur_len = len(self.cur_video_frames) - t + 1

    @property
    def shape(self):
        # type: () -> Tuple[int, int, int, int]
        """
        Returns the shape of examples being fed to the model.
        """
        return 3, 16, 256, 512

    @property
    def test_videos(self):
        # type: () -> List[str]
        """
        Returns all available test videos.
        """
        return self.test_ids

    def __len__(self):
        # type: () -> int
        """
        Returns the number of examples.
        """
        return self.cur_len

    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, torch.Tensor]
        """
        Provides the i-th example.
        """
        c, t, h, w = self.shape
        clip = self.cur_video_frames[i:i+t]
        sample = clip
        # Apply transform
        if self.transform:
            sample = self.transform(sample)
        return sample

    @property
    def collate_fn(self):
        """
        Returns a function that decides how to merge a list of examples in a batch.
        """
        return default_collate

    def __repr__(self):
        return f'ShanghaiTech (video id = {self.cur_video_id})'

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



