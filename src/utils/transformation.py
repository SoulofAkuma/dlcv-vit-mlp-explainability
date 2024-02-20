from transformers import AutoImageProcessor
from PIL import Image
from typing import List, Union
import torch

__TRANSFORMATIONS = {}

def transform_images(images: Union[Image.Image, List[Image.Image]], 
                     model: str, device=None) -> List[torch.Tensor]:
    """Perform the necessary image transformation for a given vision transformer
        on one or more images.

    Args:
        images (Union[Image.Image, List[Image.Image]]): One or more images
        model (str): The model descriptor from huggingface.co
        device (str, optional): A device to transfer the images to. Defaults to None.

    Returns:
        List[torch.Tensor]: A list of image representations as tensor
    """

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model not in __TRANSFORMATIONS:
        __TRANSFORMATIONS[model] = AutoImageProcessor.from_pretrained(model)
    if type(images) == list:
        return [__TRANSFORMATIONS[model](image, return_tensors='pt')['pixel_values'].to(device)
                for image in images]
    else:
        return __TRANSFORMATIONS[model](images, return_tensors='pt')['pixel_values'].to(device)