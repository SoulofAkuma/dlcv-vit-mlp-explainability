from transformers import AutoImageProcessor
from PIL import Image
from typing import List, Union
import torch

__TRANSFORMATIONS = {}

def transform_images(images: Union[Image.Image, List[Image.Image]], 
                     model: str, device='cpu') -> List[torch.Tensor]:
    """Perform the necessary image transformation for a given vision transformer
        on one or more images.

    Args:
        images (Union[Image.Image, List[Image.Image]]): One or more images
        model (str): The model descriptor from huggingface.co
        device (str, optional): A device to transfer the images to. Defaults to 'cpu'.

    Returns:
        List[torch.Tensor]: A list of image representations as tensor
    """
    if model not in __TRANSFORMATIONS:
        __TRANSFORMATIONS[model] = AutoImageProcessor.from_pretrained(model)
    return [transformed_img.to(device) for transformed_img in 
            __TRANSFORMATIONS[model](images, return_tensors='pt')['pixel_values']]