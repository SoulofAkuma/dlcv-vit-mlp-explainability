import torch
from typing import List, Union
from timm.models.vision_transformer import VisionTransformer
from torchvision.models.feature_extraction import create_feature_extractor

__KEY_VECTOR_EXTRACTORS = {}

def extract_value_vectors(vit: VisionTransformer, device=None) -> List[torch.Tensor]:
    """Extract the value vectors of the MLP heads of a vision transformer by block

    Args:
        vit (nn.Module): A vision transformer
    Returns:
        List[torch.Tensor]: A list of value vector matrices for each block
    """
    result = [block.mlp.fc2.weight.detach().T for block in vit.blocks]
    return result if device is None else [t.to(device) for t in result]

def extract_key_weights(vit: VisionTransformer) -> List[torch.Tensor]:
    """Extract the key vectors of the MLP heads of a vision transformer by block

    Args:
        vit (nn.Module): A vision transformer

    Returns:
        List[torch.Tensor]: A list of key vector matrices for each block
    """
    return [block.mlp.fc1.weight.detach().T for block in vit.blocks]

def extract_computed_key_vectors(vit: VisionTransformer, 
                                 images: Union[List[torch.Tensor], torch.Tensor],
                                 device: str=None) -> torch.Tensor:
    """Extract the intermediate results of the input data and the projected key vectors 
        after going through the activation function i.e. GELU(X @ W_key) for all blocks

    Args:
        vit (VisionTransformer): A vision transformer
        images (Union[List[torch.Tensor], torch.Tensor]): The preprocessed images to calculate the
            intermediate values for. If this is a list, it can either consist of individual images 
            or batches of images and will be converted to a large batch of images internally. If 
            this is a tensor, it can either be an individual image or a batch of images.
        device (str, optional): The device to 

    Returns:
        torch.Tensor: The intermediate results with the shape  (nr of blocks, nr of images 
            nr of patches, nr of value vectors/hidden dim)
    """

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if type(images) is list:
        images = torch.concat([image.unsqueeze(0) if len(image.shape) == 3 else image for image in images]).to(device)
    if len(images.shape) == 3:
        images = images.unsqueeze(0).to(device)

    extract_layers = [f'blocks.{i}.mlp.act' for i in range(len(vit.blocks))]
    if (vit, device) not in __KEY_VECTOR_EXTRACTORS:
        __KEY_VECTOR_EXTRACTORS[(vit, device)] = create_feature_extractor(vit, extract_layers).to(device)

    results = __KEY_VECTOR_EXTRACTORS[(vit, device)](images)
    return torch.stack([results[layer] for layer in extract_layers], dim=0).to(device)