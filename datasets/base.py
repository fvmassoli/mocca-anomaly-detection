from abc import ABCMeta
from abc import abstractmethod

import torch 
import numpy as np 
from torch.utils.data import Dataset


class DatasetBase(Dataset):
    """
    Base class for all datasets.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def test(self, *args):
        """
        Sets the dataset in test mode.
        """
        pass

    @property
    @abstractmethod
    def shape(self):
        """
        Returns the shape of examples.
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Returns the number of examples.
        """
        pass

    @abstractmethod
    def __getitem__(self, i):
        """
        Provides the i-th example.
        """
        pass


class OneClassDataset(DatasetBase):
    """
    Base class for all one-class classification datasets.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def val(self, *args):
        """
        Sets the dataset in validation mode.
        """
        pass

    @property
    @abstractmethod
    def test_classes(self):
        """
        Returns all test possible test classes.
        """
        pass


class VideoAnomalyDetectionDataset(DatasetBase):
    """
    Base class for all video anomaly detection datasets.
    """
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def test_videos(self):
        """
        Returns all test video ids.
        """
        pass


    @abstractmethod
    def __len__(self):
        """
        Returns the number of examples.
        """
        pass

    @property
    def raw_shape(self):
        """
        Workaround!
        """
        return self.shape

    @abstractmethod
    def __getitem__(self, i):
        """
        Provides the i-th example.
        """
        pass

    @abstractmethod
    def load_test_sequence_gt(self, video_id):
        # type: (str) -> np.ndarray
        """
        Loads the groundtruth of a test video in memory.
        :param video_id: the id of the test video for which the groundtruth has to be loaded.
        :return: the groundtruth of the video in a np.ndarray, with shape (n_frames,).
        """
        pass

    @property
    @abstractmethod
    def collate_fn(self):
        """
        Returns a function that decides how to merge a list of examples in a batch.
        """
        pass


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