import torch
from typing import List, Union

def most_predictive_vec_for_classes(values: Union[torch.Tensor, List[torch.Tensor]], 
                                    projected_values: Union[torch.Tensor, List[torch.Tensor]], device=None):
    """Get the value vectors and the indices (block, matrix col) for each class that
        best predict that class when projected

    Args:
        values (List[torch.Tensor]): The value vector matrices of the transformer either as a list
            of matrices or stacked matrices
        projected_values (Union[torch.Tensor, List[torch.Tensor]]): The projected value 
            vectors of the transformer either as a list of matrices or stacked matrices

    Returns:
        (torch.Tensor, torch.Tensor): The first tensor is a matrix of value vectors with 
            the most stimulating value vector for each class of shape (nr of classes, 
            nr of value vector features). The second is a tensor of shape (2, nr of classes)
            where the first row is the block and the second the matrix column of each of the
            provided value vectors.  
    """
    if type(projected_values) is list:
        projected_values = torch.stack(projected_values, dim=0)
        projected_values = projected_values if device is None else projected_values.to(device)
    if type(values) is list:
        values = torch.stack(values, dim=0)
        values = values if device is None else values.to(device)

    indices = most_predictive_ind_for_class(projected_values, device=device)
    return values[tuple(indices[0]), tuple(indices[1]), :], indices[:2, :]

def most_predictive_ind_for_class(projected_values: Union[torch.Tensor, List[torch.Tensor]], device=None) -> torch.Tensor:
    """Retrieve index pairs (block, matrix col, class) that contain the block and matrix column
        of the value vectors that best predicts that class when projected

    Args:
        projected_values (Union[torch.Tensor, List[torch.Tensor]]): Value vector matrices projected 
            onto the class embedding space either as list of matrices or stacked matrices

    Returns:
        torch.Tensor: Tensor of shape (3, nr_of_classes) where for every class
            the there is a block index [0], value matrix column index [1] and class index [2] 
    """
    if type(projected_values) is list:
        projected_values = torch.stack(projected_values, dim=0)

    _, _, classes = projected_values.shape
    block_repr_values, best_repr_in_block = projected_values.max(0)
    if device is not None:
        block_repr_values = block_repr_values.to(device)

    best_repr_in_value = block_repr_values.argmax(dim=0)
    class_indices = torch.arange(classes)
    if device is not None:
        best_repr_in_value = best_repr_in_value.to(device)
        class_indices = class_indices.to(device)

    result = torch.stack([best_repr_in_block[best_repr_in_value, class_indices],
                        best_repr_in_value,
                        class_indices], dim=0)
    return result if device is None else result.to(device)

def k_most_predictive_ind_for_class(projected_values: Union[torch.Tensor, List[torch.Tensor]], 
                                    k: int, device: str=None) -> torch.Tensor:
    """Get the index of the k most predictive value vectors.

    Args:
        projected_values (Union[torch.Tensor, List[torch.Tensor]]): The value vectors projected into the class embedding space.
        k (int): The top k indices will be extracted.
        device (str, optional): The device to return the vector on. Defaults to None.

    Returns:
        torch.Tensor: A tensor of shape (k, 2, nr_of_classes) that contains k pairs of block index, row index pairs for each class.
    """
    if type(projected_values) is list:
            projected_values = torch.stack(projected_values, dim=0)

    _, hidden, _ = projected_values.shape
    _, best_repr_ind = projected_values.flatten(end_dim=1).topk(k, dim=0)
    if device is not None:
        best_repr_ind = best_repr_ind.to(device)

    block_indices = best_repr_ind // hidden
    row_indices = best_repr_ind % hidden
    result = torch.stack([block_indices, row_indices]).swapaxes(0, 1)

    return result if device is None else result.to(device)