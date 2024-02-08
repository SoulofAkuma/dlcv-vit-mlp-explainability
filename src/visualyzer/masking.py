
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np

def apply_heatmap(mask, image):
    # Normalize mask and reshape
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.reshape(1, 1, 14, 14).float()
    mask = torch.nn.functional.interpolate(mask, size=224, mode='bilinear')
    mask = torch.squeeze(mask).detach().numpy()

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # Process image
    im = np.transpose(image.detach().numpy(), (1, 2, 0))
    im = (im - im.min()) / (im.max() - im.min())

    # Apply heatmap to image
    vis = heatmap + np.float32(im)
    vis = vis / np.max(vis)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    return vis

def apply_binary_mask(mask, image, percentile=80):
    # Normalize mask and reshape
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.reshape(1, 1, 14, 14).float()
    mask = torch.nn.functional.interpolate(mask, size=224, mode='bilinear')
    mask = torch.squeeze(mask).detach().numpy()

    # Determine the threshold and create binary mask
    threshold = np.percentile(mask, percentile)
    binary_mask = np.where(mask > threshold, 1, 0)

    # Process image
    im = np.transpose(image.detach().numpy(), (1, 2, 0))
    im = (im - im.min()) / (im.max() - im.min())

    # Apply binary mask to image
    masked_image = im * binary_mask[..., np.newaxis]

    return masked_image