import os
import sys
import logging

from .mvtec import MVTec_DataHolder
from .cifar10 import CIFAR10_DataHolder


AVAILABLE_DATASETS = ('cifar10', 'shanghai', 'MVTec_Anomaly')


class DataManager(object):
    """"Init class to manage and load data
    
    """
    def __init__(self, dataset_name: str, data_path: str, normal_class: int, clip_length: int=16, only_test: bool=False):
        """Init the DataManager

        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        data_path : str 
            Path to the dataset
        normal_class : int 
            Index of the normal class
        clip_length: int 
            Number of video frames in each clip (ShanghaiTech only)
        only_test : bool
            True if we are in test model, False otherwise 

        """
        self.dataset_name = dataset_name 
        self.data_path = data_path
        self.normal_class = normal_class
        self.clip_length = clip_length
        self.only_test = only_test

        # Immediately check if the data are available
        self.__check_dataset()

    def __check_dataset(self) -> None:
        """Checks if the required dataset is available
        
        """
        assert self.dataset_name in AVAILABLE_DATASETS, f"{self.dataset_name} dataset is not available"
        assert os.path.exists(self.data_path), f"{self.dataset_name} dataset is available but not found at: \n{self.data_path}"
        
    def get_data_holder(self):
        """Returns the data holder for the required dataset
        
        Rerurns
        -------
        MVTec_DataHolder : MVTec_DataHolder
            Class to handle datasets

        """
        if self.dataset_name == 'cifar10':
            return CIFAR10_DataHolder(root=self.data_path, normal_class=self.normal_class)

        if self.dataset_name == 'shanghai':
            raise NotImplementedError

        if self.dataset_name == 'MVTec_Anomaly':
            texture_classes = tuple(["carpet", "grid", "leather", "tile", "wood"])
            object_classes = tuple(["bottle", "hazelnut", "metal_nut", "screw"])
            # object_classes2 = tuple(["capsule", "toothbrush", "cable", "pill", "transistor", "zipper"])
            
            # check if the selected class is texture-type
            is_texture = self.normal_class in texture_classes
            if is_texture:
                image_size = 512
                patch_size = 64
                rotation_range = (0, 45)
            else:
                patch_size = 1
                image_size = 128
                # For some object-type classes, the anomalies are the rotations themselves
                # thus, we don't have to apply rotations as data augmentation 
                rotation_range = (-45, 45) if self.normal_class in object_classes else (0, 0) 
            
            return MVTec_DataHolder(
                                data_path=self.data_path,
                                category=self.normal_class, 
                                image_size=image_size, 
                                patch_size=patch_size, 
                                rotation_range=rotation_range, 
                                is_texture=is_texture
                            )
