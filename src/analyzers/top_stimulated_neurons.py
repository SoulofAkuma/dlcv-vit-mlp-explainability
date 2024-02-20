import numpy as np
import torch

from src.utils.extraction import extract_key_vectors
from src.utils.extraction import extract_value_vectors
from src.utils.extraction import extract_computed_key_vectors
from src.datasets.ImageNet import ImageNetDataset
from src.utils.model import embedding_projection
from src.utils.load_imgs import load_imgs_from_class_idx


def find_topk_stimulated_key_vectors(model, k: int = 1, b: int = None) -> torch.tensor:
    """
    Firstly, for each class, find the top k value vectors that are best predictive. 
    Then return the key vectors that are corresponding to these value vectors.

    Args:
        model  : Vision transformer
        k (int)
        b (int) : In which block should the key vectors be investigated, if None
                    search in all 12 blocks.
    Returns:
        (torch.tensor) : with shape (k, 1000, 768)
            k : top k, 
            1000 is the number of classes,
            768 is the dimension of each key vector
        (torch.tensor) : with shape (k, 2, 1000) or (k, 1000), the 2nd case occur when b!=None
            k : top k,
            2 because there is a block_idx and a key_idx,
            1000 is the total number of classes.
    """

    # Extract key vectors and value vectors. Each of them has shape (12, 3072, 768).
    key_vectors = extract_key_vectors(model)
    key_vectors = torch.stack(key_vectors)
    key_vectors = key_vectors.transpose(1, 2)
    value_vectors = extract_value_vectors(model)
    value_vectors = torch.stack(value_vectors)

    # Project the value vectors onto the embedding space.
    if b is None:
        # value_projected.shape (12, 3072, 1000)
        value_projected = embedding_projection(model, value_vectors)
    else:
        value_projected = embedding_projection(model, value_vectors)[b]

    if b is None:
        # logits.shape (k, 1000) ,  indices.shape (k, 1000) 
        logits = value_projected.reshape(12*3072, 1000).topk(k, dim=0).values
        indices = value_projected.reshape(12*3072, 1000).topk(k, dim=0).indices

        # block_idx.shape (k, 1000), value_idx.shape (k, 1000)
        block_idx, value_idx = np.unravel_index(indices.cpu(), (12, 3072))
        block_idx, value_idx = torch.from_numpy(block_idx), torch.from_numpy(value_idx)

        # block_idx.shape (k*1000,) , value_idx.shape (k*1000,) 
        block_idx = torch.flatten(block_idx)
        value_idx = torch.flatten(value_idx)

        # topk_indices.shape (2, k*1000)
        topk_indices = torch.stack([block_idx, value_idx])
        topk_logits = logits

        topk_key_vectors = key_vectors[topk_indices[0], topk_indices[1], :].reshape(k, 1000, 768)
        topk_indices = topk_indices.reshape(2, k, 1000).transpose(0, 1)
    else:
        # logits.shape (k, 1000) ,  indices.shape (k, 1000) 
        logits = value_projected.topk(k, dim=0).values
        indices = value_projected.topk(k, dim=0).indices
        topk_indices = indices

        # # block_idx.shape (k, 1000), value_idx.shape (k, 1000)
        # block_idx, value_idx = np.unravel_index(indices.cpu(), (12, 3072))
        # block_idx, value_idx = torch.from_numpy(block_idx), torch.from_numpy(value_idx)

        # # block_idx.shape (k*1000,) , value_idx.shape (k*1000,) 
        # block_idx = torch.flatten(block_idx)
        # value_idx = torch.flatten(value_idx)

        # # topk_indices.shape (2, k*1000)
        # topk_indices = torch.stack([block_idx, value_idx])
        # topk_logits = logits

        topk_key_vectors = key_vectors[b, topk_indices, :].reshape(k, 1000, 768)

    return topk_key_vectors, topk_indices


def compare_dot_prod(
        model,
        huggingface_model_descriptor,
        dataset: ImageNetDataset,
        topk_key_vectors: torch.Tensor, 
        topk_indices: torch.Tensor,
        class_idx: int = 0,
        k: int = 1,
        num_classes_rand: int = 20,
        num_imgs_rand: int = 4,
        batch_size: int = 4,
        seed: int = 16,
        verbose: str = False):
    """
    Fix a k-th ranked key neuron for a class. Compute average dot product between this 
    neuron and images in this class. Then proceed to compute average dot product 
    between this neuron and images from other classes. Compare the resulting 
    average dot products.

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
                                                shape (k, 2, 1000)
        k (int)                           : the rank of key vector to be investigated (take the top 1 or the top 2 etc..) (default = 1)
        num_classes_rand                  : number of random classes beside the class of interest (default = 20)
        num_imgs_rand                     : number of images per class to feed into the extractor (default = 4)
        batch_size (int)                  : how many images per forward pass to feature extractor (default = 4)
        seed (int)                        : random seed for reproducing (default = 16)
        verbose (bool)                    : logging (default = False)
    Returns:
        (list[int])   : list of class indices
        (list[float]) : the corresponding average activation values
    """

    # ================== Part 1: Loading images ========================
    num_classes = 1000
    np.random.seed(seed)

    # A random subset of class indices. (class of interest included)
    rand_class_indices = torch.from_numpy(
        np.asarray(np.random.choice(
            np.delete(np.arange(num_classes), class_idx), size=num_classes_rand, replace=False)))
    rand_class_indices = torch.concat([torch.tensor([class_idx]), rand_class_indices], dim=0)
    rand_class_indices = torch.sort(rand_class_indices).values 
    if verbose: print(f'rand_class_indices {rand_class_indices}')

    # A random subset of images in a class.
    rand_img_indices = torch.from_numpy(
        np.asarray(np.random.choice(np.arange(50), size=num_imgs_rand, replace=False)))
    rand_img_indices = rand_img_indices.to(torch.long)
    rand_img_indices = torch.sort(rand_img_indices).values
    if verbose: print(f'rand_img_indices {rand_img_indices}')

    imgs = load_imgs_from_class_idx(dataset, huggingface_model_descriptor, class_idx=rand_class_indices.tolist())
    # the shape should be : (len(rand_class_indices), len(rand_img_indices), 3, 224, 224)
    imgs = imgs[:, rand_img_indices]

    # ================== Part 2: Compute dot product + activation ========================

    # This has shape (2,), contain of a block_idx and a index for the key neuron.
    key_idx = topk_indices.transpose(1, 2)[k-1, class_idx]
    block_idx, vec_idx = key_idx[0], key_idx[1]

    num_imgs = len(rand_img_indices)
    num_batches = (num_imgs//batch_size + 1) if num_imgs%batch_size!=0 else num_imgs//batch_size
    avg_act = []
    for class_idx in range(len(rand_class_indices)):
    
        batch_cursor = 0
        list_out = []
        for b_idx in range(num_batches):

            # Extract the activation values.
            # out: shape (12, batch_size, 197, 3072)
            out = extract_computed_key_vectors(
                model, 
                imgs[class_idx, torch.arange(num_imgs)[batch_cursor: batch_cursor+batch_size]])
            
            batch_cursor += batch_size

            # This should have shape (batch_size, 12, 3072, 197)
            out = out.permute(1, 0, 3, 2)
        
            # This should be the activation values of this key neuron. shape (batch_size, 197)
            out = out[:, block_idx, vec_idx, :]

            list_out.append(out)

            if verbose and b_idx % 3 == 0:
                print(f"  Num of processed batches [{b_idx+1}/{num_batches}]")

        if verbose:
            print(f"Class idx {class_idx} -- Done.")
        
        out = torch.concat(list_out, dim=0)   # (len(rand_img_indices), 197)
        
        # Compute the average activation value over all 197 activation values per image.
        # And then the average activation value over all images.
        out = torch.mean(out, dim=1)
        out = torch.mean(out, dim=0)
        avg_act.append(out.item())

    return rand_class_indices, avg_act