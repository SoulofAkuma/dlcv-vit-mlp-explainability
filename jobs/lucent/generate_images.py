from timm import create_model
from src.visualizers.lucent import key_neuron_objective, transformer_diversity_objective, image_batch
from lucent.optvis import render
import lucent.optvis as optvis
import lucent.optvis.param as param
import numpy as np

def generate_images(model_name, image_size, thresholds, classes, device):

    model = create_model(model_name, pretrained=True).to(device).eval()

    results = {}

    for cls in classes.keys():
        block = classes[cls]['block']
        index = classes[cls]['index']

        objective = key_neuron_objective(block, index)
        div_obj = key_neuron_objective(block, index) + 1e-3 * transformer_diversity_objective(block)
        param_f = image_batch(image_size, batch_size=5, device=device)

        clear_result = render.render_vis(model, objective, thresholds=thresholds, 
                                         show_image=False, show_inline=False, device=device)
        div_result = render.render_vis(model, div_obj, param_f, thresholds=thresholds,
                                       show_image=False, show_inline=False, device=device)
        
        for i, img in enumerate(clear_result):
            results[f'{cls}_clear_{thresholds[i]}'] = (img * 255).astype(np.uint8)

        for i, imglist in enumerate(div_result):
            for ii, img in enumerate(imglist):
                results[f'{cls}_div_batch_{ii}_{thresholds[i]}'] = (img * 255).astype(np.uint8)
        
    return results