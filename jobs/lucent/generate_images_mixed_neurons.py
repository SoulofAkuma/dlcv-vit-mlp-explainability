from timm import create_model
from src.visualizers.lucent import key_neuron_objective, transformer_diversity_objective, image_batch
from lucent.optvis import render
import lucent.optvis as optvis
import lucent.optvis.param as param
from lucent.optvis import transform
import numpy as np
import torch

def generate_images(model_name, image_size, model_img_size, thresholds, classes, device):

    model = create_model(model_name, pretrained=True).to(device).eval()

    results = {}

    for cls in classes.keys():

        for img_name in classes[cls].keys():


            neurons = classes[cls][img_name]

            objective = neurons[0][2] * key_neuron_objective(neurons[0][0], neurons[0][1])
            div_obj = neurons[0][2] * key_neuron_objective(neurons[0][0], neurons[0][1]) \
                      - neurons[0][2] * 0.001 * transformer_diversity_objective(neurons[0][0])

            for i in range(1, len(neurons)):
                objective += neurons[i][2] * key_neuron_objective(neurons[i][0], neurons[i][1])
                div_obj += neurons[i][2] * key_neuron_objective(neurons[i][0], neurons[i][1]) \
                           - neurons[0][2] * 0.001 * transformer_diversity_objective(neurons[i][0])


            param_f_div = image_batch(image_size, batch_size=3, device=device)
            param_f_clear = lambda: param.image(image_size, device=device)

            transforms = transform.standard_transforms_for_device(device).copy()
            transforms.append(torch.nn.Upsample(size=model_img_size, mode='bilinear', align_corners=True))

            clear_result = render.render_vis(model, objective, param_f_clear, transforms=transforms,
                                            thresholds=thresholds, show_image=False, show_inline=False, 
                                            device=device)
            
            transforms = transform.standard_transforms_for_device(device).copy()
            transforms.append(torch.nn.Upsample(size=model_img_size, mode='bilinear', align_corners=True))
            
            div_result = render.render_vis(model, div_obj, param_f_div, transforms=transforms,
                                        thresholds=thresholds,show_image=False, show_inline=False, 
                                        device=device)
            
            for i, img in enumerate(clear_result):
                results[f'{cls}_clear_{img_name}_{thresholds[i]}'] = (img * 255).astype(np.uint8)

            for i, imglist in enumerate(div_result):
                for ii, img in enumerate(imglist):
                    results[f'{cls}_div_batch_{img_name}_{ii}_{thresholds[i]}'] = (img * 255).astype(np.uint8)

    return results