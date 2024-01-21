from typing import Union, List
import torch

def embedding_projection(vit, values: Union[List[torch.Tensor],torch.Tensor]) -> torch.Tensor:
    """Project the value vectors onto the class embedding space of the transformer

    Args:
        vit (timm.models.vision_transformer.VisionTransformer): The vision transformer
        values (List[torch.Tensor]): The list of value vector matrices

    Returns:
        List[torch.Tensor]: The projection of each of the value vector matrices
    """
    proj = vit.head.eval()
    norm = vit.norm.eval()

    if type(values) is list:
        values = torch.stack(values, dim=0)


    return proj(norm(values)).detach()