from lucent.optvis.objectives import handle_batch, wrap_objective
from lucent.optvis import param, render, transform
import torch
import torch.nn.functional as F
from typing import Callable, List
from timm.models.vision_transformer import VisionTransformer
import warnings
from src.utils.extraction import extract_value_vectors
from src.utils.model import embedding_projection
from src.analyzers. mlp_value_analyzer import most_predictive_ind_for_class
from src.utils.imagenet import get_index_for_category

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
    layer_descriptor = f'blocks_{block}_mlp_act'
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
    layer_descriptor = f'blocks_{block}_mlp_act'
    
    def inner(model):
        layer = model(layer_descriptor)
        batch, patches, hidden_dim = layer.shape
        layer = layer.permute(1, 2, 0)
        layer = F.normalize(layer, dim=1)
        corr_matrix = torch.matmul(layer.swapaxes(1, 2), layer)
        return -corr_matrix.sum()
    # def inner(model):
    #     layer = model(layer_descriptor)
    #     batch, patches, hidden_dim = layer.shape
    #     corr_matrix = torch.matmul(layer.permute(1, 0, 2), layer.permute(1, 2, 0))
    #     corr_matrix = F.normalize(corr_matrix, p=2, dim=(1,2))
    #     return -sum([sum([ (corr_matrix[i]*corr_matrix[j]).sum()
    #                       for j in range(batch) if j != i])
    #                       for i in range(batch)])
    
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

def generate_most_stimulative_for_imgnet_id(model: VisionTransformer, imagenet_id: str, 
                                            diversity: bool=False, diversity_batchsize: int=None, 
                                            device: str=None, most_predictive_inds: torch.Tensor=None,
                                            iterations: List[int]=None, div_weight: float=1e2,
                                            img_size: int=128, model_img_size: int=224):
    """Generate the most stimulative image for an imagenet class or multiple images for on class
    according to some diversity objective.

    Args:
        model (VisionTransformer): The model to generate the image(s) for.
        imagenet_id (str): The imagenet id of the class to generate the image(s) for
        diversity (bool, optional): True if a diversity objective should be used on top of the default key neuron objective, to generate a set of diverse images for one neuron maximizing their distance. Defaults to False.
        diversity_batchsize (int, optional): Only applicable if diversity is True. The number of images in the diversity batch. Defaults to None.
        device (str, optional): The device to execute calculations on. Defaults to None.
        most_predictive_inds (torch.Tensor, optional): The most predictive value vectors, if they have already been computed. Defaults to None.
        iterations (List[int], optional): The number of iterations at which you want to save the image. Should be in increasing order and the last element should be the number of iterations in total. Defaults to [500].
        div_weight (float, optional): The factor with which to weight the diversity objective together with the standard neuron objective. Defaults to 1e2.
        img_size (int, optional): The image size of the image to generate. Defaults to 128.
        model-img_size (int, optional): The image size the model expects. Defaults to 224
    """

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    iterations = iterations or [500]

    if not diversity and diversity_batchsize is not None:
        warnings.warn('Creating multiple images without a diversity objective will generate the same image multiple times!')

    if most_predictive_inds is None:
        values = extract_value_vectors(model, device)
        emb_values = embedding_projection(model, values, device)
        most_predictive_inds = most_predictive_ind_for_class(emb_values, device)

    diversity_batchsize = diversity_batchsize or 5

    block, ind, _ = most_predictive_inds[:,get_index_for_category(imagenet_id)].tolist()
    neuron_obj = key_neuron_objective(block, ind)

    transforms = transform.standard_transforms_for_device(device).copy()
    transforms.append(torch.nn.Upsample(size=model_img_size, mode='bilinear', align_corners=True))

    if diversity:
        objective = neuron_obj - div_weight * transformer_diversity_objective(block)
        param_f = image_batch(img_size, diversity_batchsize, device=device)

        return render.render_vis(model, objective, param_f, transforms=transforms, thresholds=iterations,
                                 show_image=False, show_inline=False, device=device)
    else:
        param_f = lambda: param.image(img_size, device=device)

        return render.render_vis(model, neuron_obj, param_f, transforms=transforms, thresholds=iterations, 
                                 show_image=False, show_inline=False, device=device)