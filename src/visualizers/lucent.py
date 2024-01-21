from lucent.optvis.objectives import handle_batch, wrap_objective
from lucent.optvis import param
import torch
import torch.nn.functional as F
from typing import Callable

@wrap_objective()
def key_neuron_objective(block: int, column: int, batch=None):
    """Get a lucent compatible objective to optimize towards the
    value of a key vector. Ideally you pass the row index of a 
    value vector that most predicts a class here, to optimize via the
    weights for this value vector

    Args:
        block (int): The block the key neuron/column vector is on
        column (int): The index of the column vector
        batch (int): The batch size (how many images to optimize)

    Returns:
        Callable: A lucent compatible objective to maximize a transformer neuron activation
    """
    layer_descriptor = f'blocks_{block}_mlp_fc1'
    @handle_batch(batch)
    def inner(model):
        layer = model(layer_descriptor)
        return -layer[:, :, column].mean()
    return inner

@wrap_objective()
def transformer_diversity_objective(block: int):
    """Get a lucent and transformer compatible objective to optimize
    towards the diversity in images of a batch. This objective will calculate
    a correlation matrix between the gradients of the batch and will actively
    try to decorrelate them, by punishing high covariances in a batch. This code
    is derived from the original diversity objective

    Args:
        block (int): The block to diversify in

    Returns:
        Callable: A lucent compatible objective for batch diversification
    """
    layer_descriptor = f'blocks_{block}_mlp_fc1'
    
    def inner(model):
        layer = model(layer_descriptor)
        batch, patches, hidden_dim = layer.shape
        corr_matrix = torch.matmul(layer, torch.transpose(layer, 1, 2))
        corr_matrix = F.normalize(corr_matrix, p=2, dim=(1,2))
        return -sum([sum([ (corr_matrix[i]*corr_matrix[j]).sum()
                          for j in range(batch) if j != i])
                          for i in range(batch)])
    
    return inner

def image_batch(width: int, batch_size: int, height: int=None, decorrelate=True, device=None):
    """Get a lucent compatible param_f argument that contains the image
    and its parameters to optimize. This will initialize images with the
    lucent fft method, 3 channels and decorrelated colors

    Args:
        width (int): The width of the image
        batch_size (int): The size of the batch
        height (int, optional): The height of the image. Defaults to provided width.
        decorrelate (bool, optional): A flag whether to decorrelate the colors of the image

        Returns:
            Callable: A lucent compatible param_f image batch
    """
    return lambda: param.image(width, height, batch=batch_size, decorrelate=decorrelate, device=device)