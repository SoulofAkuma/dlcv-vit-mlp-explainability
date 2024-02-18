from src.utils.IndexDataset import IndexDataset
from timm.models.vision_transformer import VisionTransformer
import torch
from src.utils.extraction import extract_value_vectors
from src.utils.model import embedding_projection
from src.analyzers.mlp_value_analyzer import most_predictive_ind_for_class
from src.utils.imagenet import get_index_for_imagenet_id
from torch.utils.data import DataLoader
from src.utils.transformation import transform_images
from src.utils.extraction import extract_computed_key_vectors
import tqdm
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Tuple

def get_topk_activating_images(model: VisionTransformer, dataset: IndexDataset, 
                               huggingface_model_descriptor: str, k: int=10, device=None,
                               show_progression: bool=True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the top k images of the dataset that have the largest key vector mean scores
    for their most predicting value vector

    Args:
        model (VisionTransformer): The transformer the key vectors are from
        dataset (IndexDataset): The dataset with the most stimulating images
        huggingface_model_descriptor (str): The huggingface model descriptor for image transformation
        k (int, optional): The number of top images to return. Defaults to 10.
        device (str, optional): The device to compute on. Defaults to None.
        show_progression (bool, optional): True if this function should loop over the dataset with a
        tqdm loop. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple where the first tensor has shape k and contains
        the indices of the most stimulating image in the dataset. The second tensor is the mean key vector
        score for the most predictive value vector and has shape len(dataset)
    """

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    weighting_mean = torch.full((len(dataset),), fill_value=-torch.inf).to(device)

    values = extract_value_vectors(model, device)
    proj_values = embedding_projection(model, values, device)
    most_pred_inds = most_predictive_ind_for_class(proj_values, device)

    loop = tqdm.tqdm(range(len(dataset))) if show_progression else range(len(dataset))

    for i in loop:
        img = dataset[i]
        tensor_img = transform_images(img['img'], huggingface_model_descriptor, device)

        layer = f'blocks.{most_pred_inds[0, img["num_idx"]]}.mlp.act'
        extractor = create_feature_extractor(model, [layer]).to(device)
        extractions = extractor(tensor_img)

        act_values = extractions[layer]
        
        mean_value = act_values[:, :, most_pred_inds[1, img['num_idx']]].mean().item()
        weighting_mean[img['dataset_index']] = mean_value
        img['img'].close()

    _, top_inds = weighting_mean.topk(k)
    return top_inds.to(device), weighting_mean
        
def correct_prediction_rate(model: VisionTransformer, dataset: IndexDataset,
                                    huggingface_model_descriptor: str, device=None,
                                    show_progression: bool=True) -> Tuple[float, torch.Tensor]:
    """Run inference for all images in the dataset and calculate the percentage how many of
    these images predicted the classes that is associated to them in the dataset.

    Args:
        model (VisionTransformer): The vision transformer to run inference on.
        dataset (IndexDataset): One of the datasets in this codebase or a dataset where each index supports `__getitem__` for 'img' and returns a PIL Image and 'num_idx' and returns the index of the correct class
        huggingface_model_descriptor (str): The huggingface model descriptor of the model for image preprocessing.
        device (str, optional): The device to run inference on. Defaults to None.
        show_progression (bool, optional): True if a tqdm range should be used. Defaults to True.

    Returns:
        Tuple[float, torch.Tensor]: The percentage how many of the predictions where correct and the actual predictions for all items in the dataset.
    """

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predicted_logits = torch.empty(len(dataset)).to(device)
    correct_logits = torch.empty(len(dataset)).to(device)

    loop = tqdm.tqdm(range(len(dataset))) if show_progression else range(len(dataset))

    for i in loop:
        img = dataset[i]
        tensor_img = transform_images(img['img'], huggingface_model_descriptor, device)
        correct_logits[i] = img['num_idx']

        prediction = model(tensor_img)
        prediction = prediction.argmax(dim=1).item()

        predicted_logits[i] = prediction

    return (predicted_logits == correct_logits).mean(dtype=torch.float32), predicted_logits 

# calculate distance to images of same class via SimCLR 