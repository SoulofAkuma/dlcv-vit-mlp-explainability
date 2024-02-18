from lucent.optvis.objectives import handle_batch, wrap_objective
from lucent.optvis import param, render, transform
import torch
import torch.nn.functional as F
from typing import Callable, List, Literal, Union
from timm.models.vision_transformer import VisionTransformer
import warnings
from src.utils.extraction import extract_value_vectors
from src.utils.model import embedding_projection
from src.analyzers. mlp_value_analyzer import most_predictive_ind_for_class, k_most_predictive_ind_for_class
from src.utils.imagenet import get_index_for_imagenet_id

@wrap_objective()
def key_neuron_objective(block: int, column: int, batch=None, before_nonlinear=True):
    """Get a lucent compatible objective to optimize towards the
    value of a key vector. Ideally you pass the row index of a 
    value vector that most predicts a class here, to optimize via the
    weights for this value vector

    Args:
        block (int): The block the key neuron/column vector is on
        column (int): The index of the column vector
        batch (int, optional): The batch size (how many images to optimize). Defaults to None.
        before_nonlinear (bool, optional): True if the activation of the first fully connected
        layer should be used and False if the activations should be used after applying the nonlinear
        activation function. Using the value after the nonlinearity degrades most, but sometimes
        leads to extremely good results, while using the activations without nonlinearity guarantees
        meaningful results that are of a good quality. Defaults to True.

    Returns:
        Callable: A lucent compatible objective to maximize a transformer neuron activation
    """
    layer_descriptor = f"blocks_{block}_mlp_{'fc1' if before_nonlinear else 'act'}"
    @handle_batch(batch)
    def inner(model):
        layer = model(layer_descriptor)
        return -layer[:, :, column].mean()
    return inner

@wrap_objective()
def transformer_diversity_objective(block: int, before_nonlinear: bool=True):
    """Get a lucent and transformer compatible objective to optimize
    towards the diversity in images of a batch. This objective will calculate
    a correlation matrix between the gradients of the batch and will actively
    try to decorrelate them, by punishing high covariances in a batch. This code
    is derived from the original diversity objective

    Args:
        block (int): The block to diversify in
        before_nonlinear (bool, optional): True if the activation value before the activation function should be used. For more info see key_neuron_objective documentation. Defaults to True.

    Returns:
        Callable: A lucent compatible objective for batch diversification
    """
    layer_descriptor = f"blocks_{block}_mlp_{'fc1' if before_nonlinear else 'act'}"
    
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
                                            diversity: bool=False, batchsize: int=1, 
                                            device: str=None, most_predictive_inds: torch.Tensor=None,
                                            iterations: List[int]=None, div_weight: float=-1e3,
                                            img_size: int=128, model_img_size: int=224,
                                            keys_before_nonlinear: bool=True):
    """Generate the most stimulative image for the top value vector of an ImageNet class, maximizing 
    the corresponding key vector while optionally enforcing a diversity objective along a batch of images.

    Args:
        model (VisionTransformer): The model to generate the image(s) for.
        imagenet_id (str): The imagenet id of the class to generate the image(s) for
        diversity (bool, optional): True if a diversity objective should be used on top of the default key neuron objective, to generate a set of diverse images for one neuron maximizing their distance. Defaults to False.
        batchsize (int, optional): Only applicable if diversity is True. The number of images in the diversity batch. Defaults to 1.
        device (str, optional): The device to execute calculations on. Defaults to None.
        most_predictive_inds (torch.Tensor, optional): The most predictive value vectors for each of the ImageNet-1k classes, if they have already been computed. Defaults to None.
        iterations (List[int], optional): The number of iterations at which you want to save the image. Should be in increasing order and the last element should be the number of iterations in total. Defaults to [500].
        div_weight (float, optional): The factor with which to weight the diversity objective together with the standard neuron objective. Defaults to 1e2.
        img_size (int, optional): The image size of the image to generate. Defaults to 128.
        model_img_size (int, optional): The image size the model expects. Defaults to 224.
        keys_before_nonlinear (bool, optional): For a detailed documentation see key_neuron_objective function documentation. Defaults to True.

    """

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    iterations = iterations or [500]

    if not diversity and batchsize > 1:
        warnings.warn('Creating multiple images without a diversity objective may generate ' +
                      'the same image multiple times!')

    if most_predictive_inds is None:
        values = extract_value_vectors(model, device)
        emb_values = embedding_projection(model, values, device)
        most_predictive_inds = most_predictive_ind_for_class(emb_values, device)

    block, ind, _ = most_predictive_inds[:,get_index_for_imagenet_id(imagenet_id)].tolist()
    neuron_objective = key_neuron_objective(block, ind, before_nonlinear=keys_before_nonlinear)
    div_objective = (key_neuron_objective(block, ind, before_nonlinear=keys_before_nonlinear) 
                     - div_weight * transformer_diversity_objective(block, keys_before_nonlinear))
    
    param_f = image_batch(img_size, batchsize, device=device) if batchsize > 1 else \
        lambda: param.image(img_size, device=device)
    
    if diversity:
        return render.render_vis(model, div_objective, param_f, thresholds=iterations, show_image=False,    
                                 show_inline=False, fixed_image_size=model_img_size, device=device)
    else:
        return render.render_vis(model, neuron_objective, param_f, thresholds=iterations, show_image=False, 
                                 show_inline=False, fixed_image_size=model_img_size, device=device)
    
weighting_schema = Literal['equal','score','softmax_score']

def generate_mixed_most_stimulative_for_imgnet_id(model: VisionTransformer, imagenet_id: str, k: int, 
                                                  diversity: bool=False, batchsize: int=1, 
                                                  device: str=None, projected_values: torch.Tensor=None,
                                                  k_most_predictive_inds: torch.Tensor=None,
                                                  iterations: List[int]=None, div_weight: float=1e-3,
                                                  img_size: int=128, model_img_size: int=224,
                                                  keys_before_nonlinear: bool=True,
                                                  weighting_schema: weighting_schema='softmax_score'):
    """Generate the most stimulative image for the top k value vectors of an ImageNet class, maximizing 
    the corresponding k key vectors while optionally enforcing a diversity objective along a batch of images.

    Args:
        model (VisionTransformer): The model to generate the image(s) for.
        imagenet_id (str): The imagenet id of the class to generate the image(s) for
        k (int): The number of top k value/key vectors to consider
        diversity (bool, optional): True if a diversity objective should be used on top of the default key neuron objective, to generate a set of diverse images for one neuron maximizing their distance. Defaults to False.
        batchsize (int, optional): Only applicable if diversity is True. The number of images in the diversity batch. Defaults to 1.
        device (str, optional): The device to execute calculations on. Defaults to None.
        projected_values (torch.Tensor, optional): The projected values if they have already been computed. These are necessary for the weighting schema. Defaults to None.
        k_most_predictive_inds (torch.Tensor, optional): The indices of the top k value vectors for each of the ImageNet-1k classes, if they have already been computed. Defaults to None.
        iterations (List[int], optional): The number of iterations at which you want to save the image. Should be in increasing order and the last element should be the number of iterations in total. Defaults to [500].
        div_weight (float, optional): The factor with which to weight the diversity objective together with the standard neuron objective. Defaults to 1e2.
        img_size (int, optional): The image size of the image to generate. Defaults to 128.
        model_img_size (int, optional): The image size the model expects. Defaults to 224.
        keys_before_nonlinear (bool, optional): For a detailed documentation see key_neuron_objective function documentation. Defaults to True.
        weighting_schema (Union[uniform_ws, score_ws], optional): The weighting schema of the top k value vectors. Can either be 'uniform' where each of the the top k vectors is weighted the same, 'score' where each of the value vectors is weighted by its normalized score for the ImageNet class and 'softmax_score' where each of the value vectors is weighted by its score normalized with the softmax function. Defaults to softmax_score.

    """

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    iterations = iterations or [500]

    if not diversity and batchsize > 1:
        warnings.warn('Creating multiple images without a diversity objective may generate ' + 
                      'the same image multiple times!')
        
    if k < 1:
        raise ValueError('k must be at least one!')

    if projected_values is None:
        projected_values = embedding_projection(model, extract_value_vectors(model, device), device)
    else:
        projected_values = projected_values.to(device)
    
    if k_most_predictive_inds is None:
        k_most_predictive_inds = k_most_predictive_ind_for_class(projected_values, k, device)
    else:
        k_most_predictive_inds = k_most_predictive_inds.to(device)

    if k_most_predictive_inds.shape[0] != k:
        raise ValueError('The provided k_most_predictive_inds must match the provided k parameter')

    cls_index = get_index_for_imagenet_id(imagenet_id)

    projected_values_topk_logits = projected_values[k_most_predictive_inds[:, 0, cls_index],
                                                    k_most_predictive_inds[:, 1, cls_index],
                                                    cls_index].to(device)
    if weighting_schema == 'equal':
        weights = torch.ones(k) / k
    elif weighting_schema == 'score':
        weights = projected_values_topk_logits / torch.sum(projected_values_topk_logits, dim=0).to(device)
    elif weighting_schema == 'softmax_score':
        weights = torch.softmax(projected_values_topk_logits, dim=0, dtype=torch.float32)

    my_key_neuron_objective = lambda i: key_neuron_objective(k_most_predictive_inds[i, 0, cls_index].item(),
                                                                   k_most_predictive_inds[i, 1, cls_index].item(),
                                                                   before_nonlinear=keys_before_nonlinear)
    my_diversity_objective = lambda i: transformer_diversity_objective(
        k_most_predictive_inds[i, 0, cls_index].item(), keys_before_nonlinear
    ) 

    neuron_objective = weights[0].item() * my_key_neuron_objective(0)
    div_objective = weights[0].item() * (my_key_neuron_objective(0) - div_weight * my_diversity_objective(0))

    for i in range(1, k):
        neuron_objective += weights[i].item() * my_key_neuron_objective(i)
        div_objective += weights[i].item() * my_key_neuron_objective(i) - div_weight * my_diversity_objective(i)

    param_f = image_batch(img_size, batchsize, device=device) if batchsize > 1 else \
        lambda: param.image(img_size, device=device)

    if diversity:
        return render.render_vis(model, div_objective, param_f, thresholds=iterations, show_image=False,
                                 show_inline=False, fixed_image_size=model_img_size, device=device)
    else:
        return render.render_vis(model, neuron_objective, param_f, thresholds=iterations, show_image=False, 
                                 show_inline=False, fixed_image_size=model_img_size, device=device)