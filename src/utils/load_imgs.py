# Trung

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from src.datasets.ImageNet import ImageNetDataset
from src.utils.transformation import transform_images
from src.utils.extraction import extract_computed_key_vectors
from typing import List, Union, Tuple
from torchvision.models.feature_extraction import create_feature_extractor


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
        torch.tensor: with shape (len(class_idx), 50, 3, 224, 224) |
            - 50 is the total number of images in each class,
            - 3 is the number of channels
            - 224 is the width and height
    """

    # Class index to imagenet_id map.
    with open('data/imagenet_class_index.json', 'r') as file:
        class_index_map = json.load(file)

    all_imgs = []
    if type(class_idx) is int:
        class_idx = [class_idx]
    for idx in class_idx:

        imagenet_id = class_index_map[str(idx)][0]
        imgs = transform_images([img['img'] for img in dataset.get_images_from_class(imagenet_id)],
                                huggingface_model_descriptor)
        imgs = torch.stack(imgs)
        all_imgs.append(imgs)

    all_imgs = torch.stack(all_imgs)
    return all_imgs


def show_img_of_class(dataset: ImageNetDataset, huggingface_model_descriptor: str, class_idx: int):
    """
    Show all 50 images of a class.

    Args:
        dataset (ImageNetDataset): Imagenet dataset
        huggingface_model_descriptor (str): model for preprocessing
        class_idx (int): class index (from 0 to 999)
    Returns:
        Plot all 50 images of the class of given index.
    """

    with open('data/imagenet_class_index.json', 'r') as file:
        class_index_map = json.load(file)

    imagenet_id = class_index_map[str(class_idx)][0]
    print(f'Class name {class_index_map[str(class_idx)][1]}')
    imgs = transform_images([img['img'] for img in dataset.get_images_from_class(imagenet_id)],
                            huggingface_model_descriptor)

    num_rows = 5
    num_cols = 10
    plt.figure(figsize=(15, (1.5)*num_rows))
    for i in range(num_rows):
        for j in range(num_cols):
            plt.subplot(num_rows, num_cols, i*num_cols+j+1)
            im = np.transpose(imgs[i*num_cols + j].numpy(), (1,2,0))
            im = (im - im.min()) / (im.max() - im.min())
            plt.axis(False)
            plt.title(i*num_cols+j)
            plt.imshow(im)

    
def show_patches(dataset: ImageNetDataset, huggingface_model_descriptor: str, class_idx: int, img_idx: int = 0):
    """
    Show the image partioned into 196 patches.

    Args:
        dataset (ImageNetDataset) : the dataset
        huggingface_model_descriptor : the model descriptor for preprocessing
        class_idx (int) : class index
        img_idx (int)   : image index
    Returns:
        Plot the image that is partitioned into 196 patches.
    """

    with open('data/imagenet_class_index.json', 'r') as file:
        class_index_map = json.load(file)

    imagenet_id = class_index_map[str(class_idx)][0]
    print(f'Class name {class_index_map[str(class_idx)][1]}')
    imgs = transform_images([img['img'] for img in dataset.get_images_from_class(imagenet_id)],
                            huggingface_model_descriptor)

    patch_size = 16

    im = imgs[img_idx]
    im = np.transpose(imgs[img_idx].numpy(), (1,2,0))
    im = (im - im.min()) / (im.max() - im.min())

    num_patches_width = 224 // patch_size
    num_patches_height = 224 // patch_size

    plt.figure(figsize=(6, 6))
    plt.tight_layout()

    n_rows = num_patches_height
    n_cols = num_patches_width
    for i in range(num_patches_height):
        for j in range(num_patches_width):
            plt.subplot(n_rows, n_cols, i*n_cols+j+1)
            plt.axis(False)
            plt.imshow(im[i*patch_size:i*patch_size+patch_size, j*patch_size:j*patch_size+patch_size])


def most_activating_images_per_neuron(
    model,
    huggingface_model_descriptor: str,
    dataset: ImageNetDataset,
    class_idx: int, 
    neuron_idx: Tuple[int], 
    imgs_indices: List[int] = [idx for idx in range(50)],
    batch_size: int = 10,
    device: str = 'cpu'
) -> List[int]:
    """
    Take a neuron and a list of images from a given class, compute the activations of the neuron on 
    every image (specifically the average activation over all image patches) and sort the images 
    descendingly according to how much they cause the neuron to activate.

    Args
        model                              : The model.
        huggingface_model_descriptor (str) : The model architecture which the image preprocessing is based on.
        dataset          (ImageNetDataset) : The dataset.
        class_idx                    (int) : The class from which the images should be fetched.
        neuron_idx                 (Tuple) : The location of the interested neuron, which is a tuple consisted of (block_idx, vec_idx)
        imgs_indices                 (int) : The given list of image indices. (for each index i: 0 <= i <= 49)
                                                (Default: [0, 1, ..., 49])
        batch_size                   (int) : How many images to be fed into the feature extractor per iteration. (Default = 10)
        device                       (str) : (Default = 'cpu')
    Return
        A descendingly sorted list of image indices.
    """

    # imgs.shape (50, 3, 224, 224)
    imgs = load_imgs_from_class_idx(dataset, huggingface_model_descriptor, class_idx).squeeze(0).to(device)
    
    # Take the interested image from the given image indices list. (len(imgs_indices), 3, 224, 224)
    imgs = imgs[imgs_indices]
    
    img_batches = torch.split(imgs, batch_size)
    acts = []
    for img_batch in img_batches:
        
        # out.shape (len(img_batch), 12, 3072, 197)
        out = extract_computed_key_vectors(model, img_batch).permute(1, 0, 3, 2).detach()
        
        block_idx, vec_idx = neuron_idx
        act = out[:, block_idx, vec_idx, :]   # shape (len(img_batch), 197)
        act = torch.mean(act, dim=1)          # shape (len(img_batch))
        acts.append(act)

    acts = torch.concat(acts)
    sorted_idx = torch.argsort(acts, descending=True)  # Sort descendingly
    imgs_indices = torch.tensor(imgs_indices)[sorted_idx].tolist()

    return imgs_indices

def sort_imgs(
        model, 
        huggingface_model_descriptor, 
        dataset: ImageNetDataset, 
        class_idx: int,
        batch_size: int = 5):
    """
    Sort the 50 images in a class descendingly based on the average-score-over-patches 
    that their hidden representations in penultimate block achieve for this class. 
    Intuitively, the top images would be most suitable for visualizing what 
    intermdidate neurons are looking at.

    Args:
        model                           : The model
        huggingface_model_descriptor    : The model descriptor for preprocessing
        datset (ImageNetDataset)        : The dataset
        class_idx (int)                 : The class index
        batch_size (int)                : Number of images to feed into the
                                            feature extractor per iteration (Default=5)
    """

    layer = ["blocks.10.mlp.fc2"]
    extractor = create_feature_extractor(model, layer)
    
    # imgs.shape (50, 3, 224, 224)
    imgs = load_imgs_from_class_idx(dataset, huggingface_model_descriptor, class_idx).squeeze(0)
    
    img_batches = torch.split(imgs, batch_size)
    all_avg_logits = []
    for img_batch in img_batches:
        
        out = extractor(img_batch)["blocks.10.mlp.fc2"]     # (batch_size, 197, 768)
        proj = model.head(model.norm(out))                  # (batch_size, 197, 1000)
        patch_logits = proj[:, :, class_idx]                # (batch_size, 197)
        avg_logits = torch.mean(patch_logits, dim=1)        # (batch_size)
        all_avg_logits.append(avg_logits)

    all_avg_logits = torch.concat(all_avg_logits)
    _, sorted_idxs = torch.sort(all_avg_logits, descending=True)
    
    return sorted_idxs.tolist()