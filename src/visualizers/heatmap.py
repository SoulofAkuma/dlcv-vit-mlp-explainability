import matplotlib.pyplot as plt
import json
import torch
import cv2 as cv
import numpy as np

from typing import List
from torchvision.models.feature_extraction import create_feature_extractor
from src.utils.transformation import transform_images
from src.utils.extraction import extract_value_vectors
from src.datasets.ImageNet import ImageNetDataset

class HeatMap:
    """
    Heat map visualization of key neuron activations.
    """
    
    def __init__(
        self, 
        model,
        huggingface_model_descriptor,
        dataset: ImageNetDataset,
        topk_indices: torch.Tensor,
        device='cpu'
    ):
        """
        Args
            model                             : Vision Transformer
            huggingface_model_descriptor(str) : Model for preprocessing
            dataset (ImageNetDataset)         : The dataset
            topk_indices (torch.tensor)       : The top k indices of the most predictive value vectors per class.
                                                shape (k, 2, 1000)
            device                            : (default='cpu')
        """

        self.device = device
        self.model = model.to(device)

        # Feature extractor
        layers = [f'blocks.{i}.mlp.act' for i in range(len(model.blocks))]
        self.extractor = create_feature_extractor(model, layers)

        # Class index map
        with open('src/data/imagenet_class_index.json', 'r') as file:
            self.class_index_map = json.load(file)

        self.dataset = dataset
        self.huggingface_model_descriptor = huggingface_model_descriptor
        self.topk_indices = topk_indices
        self.k = topk_indices.shape[0]


    def show(self, class_idx: int, k: int=1, img_indices: List[int]=[i for i in range(50)], show_predict=False, verbose=False):
        """
        Plot heat maps of activations of top k key neurons from a specific class on a number of images.

        Args:
            class_idx         (int)          : Class index of interest.
            k                 (int)          : How many top key vectors to be investigated (Default k=1)
                                                (1 <= k <= self.k)
            img_indices       (List[int])    : A list of indices of images that need to be shown. (Default=[0, 1, .., 49])
            show_predict      (bool)         : Beside the heat maps, show if the top-1-class when multiplying the value vector with 
                                                the embedding matrix E equals to the ground-truth-class. (Default=False)
        """

        # Extract the block and key vector index.
        if self.topk_indices.ndim == 3:
            indices = self.topk_indices[:k, :, class_idx]   # shape (k, 2)
            block_idx, vec_idx = indices[:, 0], indices[:, 1]
        else:
            indices = self.topk_indices[:k, class_idx]   # shape (k,)
            block_idx, vec_idx = torch.tensor([10]).repeat(k), indices

        # Fetch all 50 images and preprocess them from the given class_idx.
        imagenet_id = self.class_index_map[str(class_idx)][0]
        if verbose:
            print(f'Class name {self.class_index_map[str(class_idx)][1]}')
        imgs = transform_images([img['img'] for img in self.dataset.get_images_from_imgnet_id(imagenet_id)],
                                self.huggingface_model_descriptor)

        imgs = torch.concat(imgs, dim=0)

        if img_indices is None:
            img_indices = np.arange(num_imgs)

        plt.rcParams['image.cmap'] = 'gray'
        plt.figure(figsize=((k+1)*4, len(img_indices)*5))
        plt.tight_layout()

        num_imgs = len(img_indices)
        for i, img_index in enumerate(img_indices):

            im = np.transpose(imgs[img_index].detach().numpy(), (1,2,0))
            im = (im - im.min()) / (im.max() - im.min())
            plt.subplot(num_imgs, k+1, i*(k+1)+1)
            plt.axis(False)
            plt.imshow(im)
            plt.title(f"Original img - index: {img_index}")

            # Extract the activation maps.
            img = imgs[img_index].unsqueeze(0)
            out = self.extractor(img)
            
            for j in range(k):
                # Omit the cls token, take the key vector whose activations need to be plotted.
                act = out[f'blocks.{block_idx[j]}.mlp.act']
                act = act[0, 1:, vec_idx[j]]
                act = torch.clip(act, min=0)        # Ignore the negative activations.
            
                # Build heat map: reshape the activation map, interpolate to suitable size (224x224) and convert to a color map.
                act = act.reshape(14, 14)
                act = (act-act.min()) / (act.max()-act.min())
                mask = act.reshape(1, 1, 14, 14).float()
                mask = torch.nn.functional.interpolate(mask, size=224, mode='bilinear')
                mask = torch.squeeze(mask).detach().numpy()
                heatmap = cv.applyColorMap(np.uint8(mask*255), cv.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
       
                # Apply heat map to the original image by addition.
                vis = heatmap + np.float32(im)
                vis = vis / np.max(vis)
                vis = np.uint8(vis*255)
                vis = cv.cvtColor(vis, cv.COLOR_BGR2RGB)
                plt.subplot(num_imgs, (k+1), i*(k+1)+2+j)
                plt.axis(False)
                plt.imshow(vis)
                
                title = f"Top {j+1} neuron"
                if show_predict:
                    val_vectors = torch.stack(extract_value_vectors(self.model))  # shape (12, 3072, 768)
                    val_vector = val_vectors[block_idx[j], vec_idx[j]]            # shape (768,)
                    proj_head = self.model.head.weight                            # shape (1000, 768)
                    logits = proj_head @ val_vector                               # shape (1000,)
                    pred = torch.argmax(logits).item()
                    title += f" - P/T: [{pred}/{class_idx}] - {pred==class_idx}"
                plt.title(title)

        plt.show()