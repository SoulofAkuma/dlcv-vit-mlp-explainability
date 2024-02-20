from argparse import ArgumentParser
import os
import json
from timm import create_model
import torch
from src.utils.extraction import extract_value_vectors
from src.utils.model import embedding_projection
from src.analyzers.mlp_value_analyzer import k_most_predictive_ind_for_class
from typing import Dict, List, Tuple
import torch.nn.functional as F

MODEL = 'vit_base_patch16_384'
MODEL_IMG_SIZE = 384
IMAGE_SIZE = 384
THRESHOLDS = [750,]

def create_configs(dir: str):

    os.makedirs(dir, exist_ok=True)

    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/data/imagenet_class_index.json'))
              , 'r') as file:
        mapping = json.load(file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = create_model(MODEL, pretrained=True).to(device).eval()

    values = extract_value_vectors(model, device=device)
    embedded_values = embedding_projection(model, values, device=device).to(device)
    most_pred_inds = k_most_predictive_ind_for_class(embedded_values, 4, device=device)

    configs = []

    # 1000 / 63 = 15.625 => 16 configs for 16 gpus
    for i in range(0, 1000, 63):
        # name of the generation, list of block index, row index, weight
        imagenet_classes: Dict[str, List[Tuple[int, int, float]]] = {}

        # 64 classes per gpu
        for ii in range(0, 63):
            if i + ii >= 1000:
                break
            inds_for_cls = most_pred_inds[:,:,i+ii].tolist()
            cls_embed_values = embedded_values[most_pred_inds[:,0,i+ii], most_pred_inds[:,1,i+ii],i+ii]
            weights_1_2 = torch.softmax(cls_embed_values[:2], dim=0, dtype=torch.float32)
            weights_1_2_3 = torch.softmax(cls_embed_values[:3], dim=0, dtype=torch.float32)
            weights_1_2_3_4 = torch.softmax(cls_embed_values, dim=0, dtype=torch.float32)
            imagenet_classes[mapping[str(i + ii)][0]] = {
                'top_1': [(inds_for_cls[0][0],inds_for_cls[0][1], 1.)],
                'top_1,2': [(inds_for_cls[0][0],inds_for_cls[0][1], 0.5),
                            (inds_for_cls[1][0],inds_for_cls[1][1], 0.5)],
                'top_1,2_weighted': [(inds_for_cls[0][0],inds_for_cls[0][1],weights_1_2[0].item()),
                                     (inds_for_cls[1][0],inds_for_cls[1][1],weights_1_2[1].item())],
                'top_1,2,3': [(inds_for_cls[0][0],inds_for_cls[0][1], 1/3),
                              (inds_for_cls[1][0],inds_for_cls[1][1], 1/3),
                              (inds_for_cls[2][0],inds_for_cls[2][1], 1/3)],
                'top_1,2,3_weighted': [(inds_for_cls[0][0],inds_for_cls[0][1],weights_1_2_3[0].item()),
                                       (inds_for_cls[1][0],inds_for_cls[1][1],weights_1_2_3[1].item()),
                                       (inds_for_cls[2][0],inds_for_cls[2][1],weights_1_2_3[2].item())],
                'top_1,2,3,4': [(inds_for_cls[0][0],inds_for_cls[0][1], 0.25),
                                (inds_for_cls[1][0],inds_for_cls[1][1], 0.25),
                                (inds_for_cls[2][0],inds_for_cls[2][1], 0.25),
                                (inds_for_cls[3][0],inds_for_cls[3][1], 0.25)],
                'top_1,2,3,4_weighted': [(inds_for_cls[0][0],inds_for_cls[0][1],weights_1_2_3_4[0].item()),
                                         (inds_for_cls[1][0],inds_for_cls[1][1],weights_1_2_3_4[1].item()),
                                         (inds_for_cls[2][0],inds_for_cls[2][1],weights_1_2_3_4[2].item()),
                                         (inds_for_cls[3][0],inds_for_cls[3][1],weights_1_2_3_4[3].item())],
            }

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