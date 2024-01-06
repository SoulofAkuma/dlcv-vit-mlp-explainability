import numpy as np
import torch

from src.utils.extraction import extract_key_vectors
from src.utils.extraction import extract_value_vectors
from src.utils.extraction import extract_computed_key_vectors
from src.datasets.ImageNet import ImageNetDataset
from src.utils.model import embedding_projection
from src.utils.load_imgs import load_imgs_from_class_idx


def find_topk_stimulated_key_vectors(model, k: int = 1) -> torch.tensor:
    """
    Firstly, for each class, find the top k value vectors that are best predictive. 
    Then return the key vectors that are corresponding to these value vectors.

    Args:
        model  : Vision transformer
        k (int)
    Returns:
        torch.tensor : with shape (k, 1000, 768) | 
            - 1000 is the number of classes
            - 768 is the dimension of each key vector
    """

    # Extract key vectors and value vectors. Each of them has shape (12, 3072, 768).
    key_vectors = extract_key_vectors(model)
    key_vectors = torch.stack(key_vectors)
    key_vectors = key_vectors.transpose(1, 2)
    value_vectors = extract_value_vectors(model)
    value_vectors = torch.stack(value_vectors)

    # Project the value vectors onto the embedding space.
    # value_projected.shape (12, 3072, 1000)
    value_projected = embedding_projection(model, value_vectors)

    # logits.shape (5, 1000) ,  indices.shape (5, 1000) 
    logits = value_projected.reshape(12*3072, 1000).topk(k, dim=0).values
    indices = value_projected.reshape(12*3072, 1000).topk(k, dim=0).indices

    # block_idx.shape (5, 1000), value_idx.shape (5, 1000)
    block_idx, value_idx = np.unravel_index(indices.cpu(), (12, 3072))
    block_idx, value_idx = torch.from_numpy(block_idx), torch.from_numpy(value_idx)

    # block_idx.shape (5000,) , value_idx.shape (5000,) 
    block_idx = torch.flatten(block_idx)
    value_idx = torch.flatten(value_idx)

    # topk_indices.shape (2, 5000)
    topk_indices = torch.stack([block_idx, value_idx])
    topk_logits = logits

    topk_key_vectors = key_vectors[topk_indices[0], topk_indices[1], :].reshape(5, 1000, 768)

    return topk_key_vectors, topk_indices

def compare_dot_prod(
        model,
        huggingface_model_descriptor,
        dataset: ImageNetDataset,
        topk_key_vectors: torch.tensor, 
        topk_indices: torch.tensor,
        class_idx: int = 0,
        k: int = 1,
        num_classes_rand: int = 20,
        num_imgs_rand: int = 4,
        seed: int = 16):
    """
    Find avg dot product for images not from class best predicted by neuron as a baseline activation for that value vector
    Find avg dot product for image from the class best predicted by the neuron and compare with baseline.
    Here beside the dot product the activation of GELU will also be applied.

    Because of limited computation power only use a random subset of classes and 
    for each class only use a random subset of images.

    Args:
        model                             : Vision Transformer
        huggingface_model_descriptor(str) : model for preprocessing
        dataset (ImageNetDataset)         : the dataset
        class_idx (int)                   : index of class of interest (default = 0)
        topk_key_vectors (torch.tensor)   : the top k key vectors corresponding to the top k most predictive value vectors per class.
                                                shape (k, 1000, 768)
        topk_indices (torch.tensor)       : the top k indices of the most predictive value vectors per class.
                                                shape (2, 1000)
        k (int)                           : the index of key vector to be investigated (take the top 1 or the top 2 etc..) (default = 1)
        num_classes_rand                  : number of random classes beside the class of interest (default = 20)
        num_imgs_rand                     : number of images per class to feed into the extractor (default = 4)
        seed (int)                        : random seed for reproducing (default = 16)
    Returns:
        (list[int])   : list of class indices
        (list[float]) : the corresponding average activation values
    """

    # ================== Part 1: Loading images ========================
    num_classes = 1000
    np.random.seed(seed)

    # Random subset of class indices. (class of interest included)
    rand_class_indices = torch.from_numpy(
        np.asarray(np.random.choice(np.delete(np.arange(num_classes), class_idx), size=num_classes_rand)))
    rand_class_indices = torch.concat([torch.tensor([class_idx]), rand_class_indices], dim=0)
    rand_class_indices = torch.sort(rand_class_indices).values 
    print(f'rand_class_indices {rand_class_indices}')

    # Random subset of images in a class.
    rand_img_indices = torch.from_numpy(
        np.asarray(np.random.choice(np.arange(50), size=num_imgs_rand, replace=False)))
    rand_img_indices = rand_img_indices.to(torch.long)
    rand_img_indices = torch.sort(rand_img_indices).values
    print(f'rand_img_indices {rand_img_indices}')

    imgs = load_imgs_from_class_idx(dataset, huggingface_model_descriptor, class_idx=rand_class_indices.tolist())
    # the shape should be : (len(rand_class_indices), len(rand_img_indices), 3, 224, 224)
    imgs = imgs[:, rand_img_indices]

    # ================== Part 2: Compute dot product + activation ========================

    # Fetch the desired key vector and its index (block_idx, key_idx).
    key_vec = topk_key_vectors[k, class_idx]
    key_idx = topk_indices.T[class_idx]

    avg_act = []
    for i, idx in enumerate(rand_class_indices):
        #print(f'class index = {idx}')
        #print(f'rand img indices = {rand_img_indices.tolist()}')
    
        # Extract the activation values.
        # out: shape (len(rand_img_indices), 12, 197, 3072)
        out = extract_computed_key_vectors(model, imgs[i])

        # This should have shape (len(rand_img_indices), 12, 3072, 197)
        out = out.permute(1, 0, 3, 2)
        
        # This should be the activation values of this key neuron. shape (len(rand_img_indices), 197)
        out = out[:, key_idx[0], key_idx[1], :]
        
        # Compute the average activation value over all 197 activation values per image.
        # And then the average activation value over all images.
        out = torch.mean(out, dim=1)
        out = torch.mean(out, dim=0)
        avg_act.append(out.item())


    return rand_class_indices, avg_act

