# Trung

import torch
import json
from src.datasets.ImageNet import ImageNetDataset
from src.utils.transformation import transform_images
from typing import List, Union


def load_imgs_from_class_idx(
        dataset: ImageNetDataset,
        huggingface_model_descriptor: str,
        class_idx: Union[List[int], int]
        ) -> torch.tensor:
    """
    Get all images from the given class index or list of class indices. The
    images will also be preprocessed.

    Args:
        dataset (ImageNetDataset)          : the dataset
        huggingface_model_descriptor (str) : the model configuration that the preprocessing based on
        class_idx (Union[List[int], int])  : can either be a single integer or a python list of integers.
    Returns:
        torch.tensor: with shape (len(class_idx), 50, 3, 244, 244) |
            - 50 is the total number of images in each class,
            - 3 is the number of channels
            - 244 is the width and height
    """

    # Class index to imagenet_id map.
    with open('../data/imagenet_class_index.json', 'r') as file:
        class_index_map = json.load(file)

    all_imgs = []
    for idx in class_idx:

        imagenet_id = class_index_map[str(idx)][0]
        imgs = transform_images([img['img'] for img in dataset.get_images_from_imgnet_id(imagenet_id)],
                                huggingface_model_descriptor)
        imgs = torch.stack(imgs)
        all_imgs.append(imgs)

    all_imgs = torch.stack(all_imgs)
    return all_imgs