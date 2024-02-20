from argparse import ArgumentParser
import os
import json
from timm import create_model
import torch
from src.utils.extraction import extract_value_vectors
from src.utils.model import embedding_projection
from src.analyzers.mlp_value_analyzer import most_predictive_ind_for_class

MODEL = 'vit_base_patch16_384'
MODEL_IMG_SIZE = 384
IMAGE_SIZE = 384
THRESHOLDS = [250, 500, 750, 1000]

def create_configs(dir: str):

    os.makedirs(dir, exist_ok=True)

    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/data/imagenet_class_index.json'))
              , 'r') as file:
        mapping = json.load(file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = create_model(MODEL, pretrained=True).to(device).eval()

    values = extract_value_vectors(model, device=device)
    embedded_values = embedding_projection(model, values, device=device).to(device)
    most_pred_inds = most_predictive_ind_for_class(embedded_values, device=device)

    configs = []

    for i in range(0, 1000, 63):
        imagenet_classes = {}
        for ii in range(0, 63):
            if i + ii >= 1000:
                break
            block, ind, _ = most_pred_inds[:,i+ii].tolist()
            imagenet_classes[mapping[str(i + ii)][0]] = {'block': block, 'index': ind}
        configs.append({
                'model': MODEL,
                'model_img_size': MODEL_IMG_SIZE,
                'classes': imagenet_classes,
                'image_size': IMAGE_SIZE,
                'thresholds': THRESHOLDS,
            })

    for i, config in enumerate(configs):
        with open(os.path.join(dir, f'config_{i}.json'), 'w+') as file:
            json.dump(config, file)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output-dir', type=str)

    args = parser.parse_args()
    create_configs(os.path.abspath(args.output_dir))