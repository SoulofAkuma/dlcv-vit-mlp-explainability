import torch
from typing import List, Union

def most_predictive_vec_for_classes(values: Union[torch.Tensor, List[torch.Tensor]], 
                                    projected_values: Union[torch.Tensor, List[torch.Tensor]]):
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
    if type(values) is list:
        values = torch.stack(values, dim=0)

    indices = most_predictive_ind_for_class(projected_values)
    return values[tuple(indices[0]), tuple(indices[1]), :], indices[:2, :]

def most_predictive_ind_for_class(projected_values: Union[torch.Tensor, List[torch.Tensor]]):
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
    best_repr_in_value = block_repr_values.argmax(dim=0)
    class_indices = torch.arange(classes)
    return torch.stack([best_repr_in_block[best_repr_in_value, class_indices],
                        best_repr_in_value,
                        class_indices], dim=0)