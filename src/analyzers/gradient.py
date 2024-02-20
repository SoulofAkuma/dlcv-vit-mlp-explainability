import matplotlib.pyplot as plt
import json
import torch
import cv2 as cv
import numpy as np

from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from typing import List
from torchvision.models.feature_extraction import create_feature_extractor
from src.utils.transformation import transform_images
from src.utils.extraction import extract_value_vectors
from src.datasets.ImageNet import ImageNetDataset

class Gradient:
    """
    Gradient of key neuron activations with respect to the input image.
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
        layer_types = ["mlp.act", "mlp.fc2", "add_1"]
        layers = [f"blocks.{b}.{lt}" for lt in layer_types for b in range(12)]
        self.extractor = create_feature_extractor(model, layers)

        # Class index map
        with open('data/imagenet_class_index.json', 'r') as file:
            self.class_index_map = json.load(file)

        self.dataset = dataset
        self.huggingface_model_descriptor = huggingface_model_descriptor
        self.topk_indices = topk_indices
        self.k = topk_indices.shape[0]

    def show(
        self,
        class_idx: int,
        num_patches: int = 3,
        img_indices: List[int]=[i for i in range(10)],
        agg: bool = False,
        agg_param: float = 0.5,
        num_agg: int = 3,
        plot: bool = True,
        verbose: bool = True
    ):
        """
        A gradient approach that use gradient value of neural activation in a single patch 
        with respect to the image pixel to highlight the object of interest in the image.

        For the given class, take the top 1 neuron and look at the image patches that most predictive for
        this class. Plot the gradient of the neural activation in these patches with respect to the 
        input image. 
        
        Result: In most of the cases, the pixels belonging to the object of interest will
        usually have high positive gradient. Therefore the gradient maps will form a shape
        that is similiar to the shape object of interest. It means that the pixels
        belonging to the object are important for the neural activations, and hence are 
        crucial for the prediction.

        If agg is True, then the sum all 196 gradient maps with an exponential
        decaying weights. This will make the gradients of the pixels belonging
        to the object of interest even more larger, thus improve the visualization.
        But also increase gradient noise.

        Args:
            class_idx             (int)  : class of interest
            num_patches           (int)  : number of image patches (Default = 3)
            img_indices     (List[int])  : list of image indices   (Default [0, 1, 2, ..., 9])
            agg                  (bool)  : Default=False, if True, then beside the gradient of neural activation
                        of single patch, also compute the gradient of an exponential sum of the top num_agg patches
                        with parameter agg_param This should enhance the gradient visualization.
            agg_param            (float) : (Default=0.5)
            num_agg              (int)   : (Default=3) 
        """

        grads_maps = []

        use_heatmap = True if num_patches is None else False
        num_patches = 2 if num_patches is None else num_patches

        if agg and not use_heatmap:
            num_patches += 1

        # Extract the block and key vector index of top 1 neuron.
        if self.topk_indices.ndim == 3:
            indices = self.topk_indices[0, :, class_idx]   # shape (2,)
            block_idx, vec_idx = indices[0], indices[1]
        else:
            indices = self.topk_indices[0, class_idx]   # shape (1,)
            block_idx, vec_idx = torch.tensor(10), indices

        if verbose: print(f"block_idx, vec_idx = {block_idx.item(), vec_idx.item()}")

        # Fetch all 50 images and preprocess them from the given class_idx.
        imagenet_id = self.class_index_map[str(class_idx)][0]
        if verbose: print(f'Class name {self.class_index_map[str(class_idx)][1]}')
        imgs = transform_images([img['img'] for img in self.dataset.get_images_from_class(imagenet_id)],
                                self.huggingface_model_descriptor)

        num_imgs = len(img_indices)
        
        if plot:
            plt.figure(figsize=(4*(num_patches+1), 4*(num_imgs)))
            plt.tight_layout()

        for i, idx in enumerate(img_indices):
            
            img = imgs[idx].unsqueeze(0)
            img.requires_grad_()

            out = self.extractor(img)
            mlp_act = out[f"blocks.{block_idx}.mlp.act"]   # shape (1, 197, 3072)
            mlp_fc2 = out[f"blocks.{block_idx}.mlp.fc2"]   # shape (1, 197, 768)

            # Look at the activations of this top 1 neuron on 196 patches excluding cls patch (shape: 196).
            key_act = mlp_act[0, 1:, vec_idx]

            # Sort the patch descendingly according to how much they cause the neuron to activate.
            top_activated_patches = torch.argsort(key_act, descending=True)

            pred = self.model.head(self.model.norm(mlp_fc2))
            ordered = torch.argsort(pred[0, 1:, class_idx], descending=True)
            top_activated_patches = ordered

            im = img.squeeze(0).permute(1,2,0).detach().numpy()
            im = (im-im.min()) / (im.max()-im.min())
            
            if plot:
                plt.subplot(num_imgs, num_patches+1, i*(num_patches+1)+1)
                plt.axis(False)
                plt.title(f"Original img - index: {idx}")
                plt.imshow(im)

            if use_heatmap:
                
                # Sum of 196 gradient maps.
                # The gradient map belonging to the patch that has high correct class prediction score
                # will have more weights.
                weights = [(agg_param)**m for m in range(num_agg)]
                objective = sum([weights[n]*key_act[top_activated_patches[n]] for n in range(num_agg)])

                grads = torch.autograd.grad(objective, img, retain_graph=True)
                grads = grads[0]
                grads = grads.squeeze(0).permute(1,2,0).detach()
                grads = np.clip(grads, a_min=0, a_max=1)

                # ---- Start convert this gradient-map-sum to heat map --------
                # The steps are:
                # 1. Partition the gradient map into non-overlapping patches of size 16x16.
                # 2. Compute the mean color in each subwindow.
                # 3. Use interpolation to resize it to the original input size.
                # 4. Convert to heat map.
                    
                patch_size = 16
                num_patches_width = 224 // patch_size
                num_patches_height = 224 // patch_size

                # Initialize the patches
                patches = torch.zeros(num_patches_height, num_patches_width, 3)

                # Compute mean color in each patch.
                for i_ in range(num_patches_height):
                    for j in range(num_patches_width):
                        patch = grads[i_*patch_size:i_*patch_size+patch_size, j*patch_size:j*patch_size+patch_size]
                        avg_color = torch.mean(patch, dim=(0, 1))
                        patches[i_, j] = avg_color

                # Convert the patches to grayscale and use interpolation to resize it to the original input size.
                patches = cv.cvtColor(patches.numpy(), cv.COLOR_RGB2GRAY)
                patches = torch.from_numpy(patches)
                patches = (patches-patches.min()) / (patches.max()-patches.min())
                patches = patches.reshape(1, 1, num_patches_height, num_patches_width).float()
                patches = torch.nn.functional.interpolate(patches, size=224, mode='bilinear')
                patches = torch.squeeze(patches).numpy()
                
                # Convert to a heatmap.
                heatmap = cv.cvtColor(cv.applyColorMap(np.uint8(patches*255), cv.COLORMAP_JET), cv.COLOR_BGR2RGB)
                #print(f"heatmap.shape {heatmap.shape}")
                #print(f"im.shape {im.shape}")
                heatmap = np.float32(heatmap) / 255
                
                if plot:
                    plt.subplot(num_imgs, num_patches+1, i*(num_patches+1)+2)
                    plt.axis(False)
                    plt.imshow(heatmap)
                    plt.title("Aggregated gradient")

                # Apply the heatmap
                applied = np.float32(im) + heatmap
                applied = applied / np.max(applied)
                if plot:
                    plt.subplot(num_imgs, num_patches+1, i*(num_patches+1)+3)
                    plt.axis(False)
                    plt.imshow(applied)
                    plt.title("Applied")
                
            else:
                
                # NOT IMPORTANT - IGNORE PLEASE.
                # for j in range(num_patches):

                #     if agg and (j == num_patches-1):
                #         weights = [(agg_param)**m for m in range(num_agg)]
                #         objective = sum([weights[n]*key_act[top_activated_patches[n]] for n in range(num_agg)])
                #     else:
                #         objective = key_act[top_activated_patches[j]]
                    
                #     grads = torch.autograd.grad(objective, img, retain_graph=True)
                #     grads = grads[0]
                #     grads = grads.squeeze(0).permute(1,2,0).detach()

                #     grads = np.clip(grads, a_min=0, a_max=1)
                #     #grads = cv.GaussianBlur(grads.numpy(), ksize=(3,3), sigmaX=0)
                    
                #     plt.subplot(num_imgs, num_patches+1, i*(num_patches+1)+j+2)
                #     plt.axis(False)
                #     if agg and (j == num_patches-1):
                #         plt.title(f"Sum ({agg_param:.2f}) of top {num_agg}")
                #     else:
                #         plt.title(f"Top {j+1} predictive patch")

                #     grads_maps.append(grads)
                    
                    #grads = cv.cvtColor(grads.numpy(), cv.COLOR_RGB2GRAY)
                    #grads = cv.cvtColor(cv.applyColorMap(np.uint8(grads.numpy()*255), cv.COLORMAP_JET), cv.COLOR_BGR2RGB)
                    # plt.imshow(grads)  
                pass          

        if plot:  
            plt.show()

        return im, heatmap, applied

    def useless_gradient(
        self,
        class_idx,
        img_idx: int,
        num: int = 1,
        block_idx: int = 10
    ):
        """
        Another gradient method for highlighting the object of interest in an image.
        In this case, the gradient of the prediction loss with respect to the
        neural activation are used.
        
        The essential ideas are:

        1. Take a neuron X that is not best for predicting a class A.
        This neuron doesn't focus on the relevant part of the object in the
        image. When we take it's hidden representation in the penultimate 
        block (block 11) and project it to the embedding space, the scores for 
        class A are very low and hence it's losses are large.

        2. Compute the gradient of these losses with respect to the 196 neural
        activations of this neuron X, because there are in total 196 image 
        patches and each patch responsible for a loss.

        3. Reshape these 196 neural activations into (14, 14) and plot a
        upsized heatmap based on the activation. If succeed, the gradients
        will somewhat highlight the object.

        4. If this is the case, these gradients will act sort of an 
        "advice signal", which imply: Had this neuron X focused on these 
        highlighted patches instead, it's loss would have decreased and it 
        would have promoted the class better.

        Disclaimer: Unfortunately to find this specific low-ranked neuron X
        is not easy and not every low-ranked neuron admit such gradients
        that highlight the relevant patches. Most of the time the gradients
        would highlight random or completely irrelevant patches.

        However, these kind of low-ranked neurons X usually have a high
        average-neural-activations across the patches. So a proper way to
        identify them would be investigate the top neurons having the highest
        average-neural-activations across patches. For each of these 
        investigated neuron, this function would plot their gradient map and
        it's activations (both as heatmap) for visualization.

        This function only serves as a method for identifying these neuron X.

        Args:
            class_idx (int)			: class of interest
            img_idx	(int)			: index of image
            num (int)				: number of neurons to be investigated (default=4)
            block_idx (int)			: the block which contains the neurons
                                        that need to be investigated (default=10)
        """

        # ------- Define extractor and load image --------------
        layer_types = ["mlp.act", "mlp.fc2", "add_1"]
        layers = [f"blocks.{b}.{lt}" for lt in layer_types for b in range(12)]
        extractor = create_feature_extractor(self.model, layers)
        block_idx = 10
        with open('data/imagenet_class_index.json', 'r') as file:
            class_index_map = json.load(file)

        print(f'Class name {class_index_map[str(class_idx)][1]}')
        imagenet_id = class_index_map[str(class_idx)][0]
        imgs = transform_images([img['img'] for img in self.dataset.get_images_from_class(imagenet_id)],
                        self.huggingface_model_descriptor)
        img = imgs[img_idx].unsqueeze(0)

        # 1. Feed the image into extractor.
        # 2. Extract mlp activations and features from fc2.
        # 3. Project fc2 into class embedding.
        # 4. Every patch promote some class, compute the loss occured by each patch
        #		with respect to the neural activations of this patch (3072 neurons)
        # 5. Only investigate the gradient with respect to the neuron
        #		having the largest average-activation across patches.
        
        img.requires_grad_()
        out = extractor(img)
        mlp_act = out[f"blocks.{block_idx}.mlp.act"]		        # (1, 197, 3072)
        mlp_fc2 = out[f"blocks.{block_idx}.mlp.fc2"]		        # (1, 197, 768)

        # Project hidden representations into embedding space.
        res_proj = self.model.head(self.model.norm(mlp_fc2))		# (1, 197, 1000)

        # Compute rank of value vector based on how predictive they are for each class.
        value_vectors = torch.stack(
            extract_value_vectors(self.model))[block_idx]		    # (3072, 768)
        val_proj = self.model.head(self.model.norm(value_vectors))	# (3072, 1000)
        val_proj = torch.argsort(
            val_proj, dim=0, descending=True)				        # (3072, 1000)	

        # Compute cross entropy loss
        num_classes = 1000
        ce = CrossEntropyLoss(reduction='none')
        res_proj = res_proj.squeeze(0)
        res_proj = softmax(res_proj, dim=1)

        target = torch.zeros(1, num_classes).to(self.device)
        target[0, class_idx] = 1
        target = target.repeat(res_proj.shape[0], 1) 
        loss = ce(res_proj, target)

        # Average activation values of 3072 neurons and sort them descendingly.
        avg_act = torch.mean(mlp_act[0, 1:], dim=0)
        ordered_act_indices = torch.argsort(avg_act, descending=True)

        # ------ Compute gradient of loss (shape 197) with respect to the neural activation ------.
        large_grads = torch.zeros(196, 3072)
        grad_map = torch.zeros(196)
        for j in range(1, 197):
            grads = torch.autograd.grad(loss[j], mlp_act, retain_graph=True)	
            grads = -grads[0][0, j]			# (3072,)
            large_grads[j-1] = grads

        # ----------------------- Start plotting here -----------------------------
        ncols = 4
        plt.figure(figsize=(4*ncols, 4*num))
        plt.tight_layout()
        for i in range(num):
            grad_map = torch.zeros(196)
            for j in range(1, 197):
                grad_map[j-1] += large_grads[j-1, ordered_act_indices[i]]
                    
            # Only allow positive gradients in gradient map.
            # Without this step it wouldn't work. Probably because a gradient map
            # that is dominated by negative gradients will cause the heat map to
            # be inverted.
            # Because negative gradient would only increase the loss. So applying
            # relu to eliminate them would also be sensible.
            grad_map = torch.nn.functional.relu(grad_map)

            im = img.squeeze(0).permute(1,2,0).detach().numpy()
            im = (im-im.min()) / (im.max()-im.min())
            plt.subplot(num, ncols, i*ncols+1)
            plt.axis(False)
            plt.title(f"Original img - index {img_idx}")
            plt.imshow(im)

            # ------- Compute and plot gradient heat map -------------
            grad_map = (grad_map - grad_map.min()) / (grad_map.max() - grad_map.min())
            grad_map = grad_map.reshape(1, 1, 14, 14).float()
            grad_map = torch.nn.functional.interpolate(grad_map, size=224, mode='bilinear')
            grad_map = torch.squeeze(grad_map).detach().numpy()
            heatmap = cv.applyColorMap(np.uint8(255 * grad_map), cv.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255

            plt.subplot(num, ncols, i*ncols+2)
            plt.axis(False)
            plt.title(f"Gradient ({i+1})")
            plt.xlabel("Img patch index")
            plt.ylabel("Gradient")
            plt.imshow(cv.cvtColor(heatmap, cv.COLOR_BGR2RGB))

            # -------- Apply headmap to image --------------------------
            vis = heatmap + np.float32(im)
            vis = vis / np.max(vis)
            vis = np.uint8(255 * vis)
            vis = cv.cvtColor(np.array(vis), cv.COLOR_RGB2BGR)
            plt.subplot(num, ncols, i*ncols+3)
            plt.axis(False)
            plt.title("Gradient Applied")
            plt.imshow(vis)

            # Omit the cls token, take the key vector whose activations need to be plotted.
            act = mlp_act[0, 1:, ordered_act_indices[i]]

            # Build heat map: reshape the activation map, interpolate to suitable size (224x224) and convert to a color map.
            act = act.reshape(14, 14)
            act = (act-act.min()) / (act.max()-act.min())
            mask = act.reshape(1, 1, 14, 14).float()
            mask = torch.nn.functional.interpolate(mask, size=224, mode='bilinear')
            mask = torch.squeeze(mask).detach().numpy()
            heatmap = cv.applyColorMap(np.uint8(mask*255), cv.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255

            # Compute the rank of this neuron.
            rank = (val_proj[:, class_idx] == ordered_act_indices[i]).nonzero().item()

            # Apply heat map to the original image by addition.
            vis = heatmap + np.float32(im)
            vis = vis / np.max(vis)
            vis = np.uint8(vis*255)
            vis = cv.cvtColor(vis, cv.COLOR_BGR2RGB)
            plt.subplot(num, ncols, i*ncols+4)
            plt.axis(False)
            plt.title(f"Activation (Rank: {rank+1})")
            plt.imshow(vis)
        
        plt.show()