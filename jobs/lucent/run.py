import os
import json
import torch
from generate_images import generate_images
from PIL import Image
import time

CONFIG_PATH = '/scratch/vihps/vihps01/vit-mlp-explainability/configs'
RESULTS_PATH = '/scratch/vihps/vihps01/vit-mlp-explainability/images'
RESULT_STATS_PATH = '/scratch/vihps/vihps01/vit-mlp-explainability/job-reports'

os.makedirs(CONFIG_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(RESULT_STATS_PATH, exist_ok=True)

job_index = os.environ['SLURM_PROCID']
node_name = os.environ['SLURMD_NODENAME']

config = None
with open(os.path.join(CONFIG_PATH, f'config_{job_index}.json'), 'r') as file:
    config = json.load(file)

device_count = torch.cuda.device_count()
device = torch.device(f'cuda:{int(job_index) % device_count}'
                      if torch.cuda.is_available() else 'cpu')


images = generate_images(config['model'], config['image_size'], config['thresholds'], config['classes'])

start_time = time.time()
for img_name in images.keys():
    Image.fromarray(images[img_name]).save(os.path.join(RESULTS_PATH, f'{img_name}.png'))

with open(os.path.join(RESULT_STATS_PATH, f'results_{job_index}.json'), 'w+') as file:
    json.dump({
        'job_index': job_index,
        'node': node_name,
        'execution_time': time.time() - start_time,
        'images_generated': images.keys()
    }, file)