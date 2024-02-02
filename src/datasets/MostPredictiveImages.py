from src.utils.IndexDataset import IndexDataset
import os

import pandas as pd
from PIL import Image
from typing import Literal, Set, Union
from src.utils.imagenet import MAPPING_FRAME

iter_T = Literal[250, 500, 750]
types_T = Literal['div', 'clear']
batch_ind_T = Literal[0,1,2,3]

class MostPredictiveImagesDataset(IndexDataset):

    def __init__(self, dataset_path: str, gen_types: Union[Literal['all'], types_T],
                 iterations: Set[iter_T]):
        super().__init__()
        self.dataset_path = dataset_path
        self.gen_type  = gen_types
        self.iterations = list(iterations)
        
        self.typeFactor = {'all': 6, 'div': 5, 'clear': 1}[self.gen_type]
        self.length = len(MAPPING_FRAME.index) * self.typeFactor * len(iterations)

    def get_images_from_imgnet_id(self, imagenet_id: str):
        cls_ind = MAPPING_FRAME.loc[imagenet_id]['num_idx']
        return [self[index] for index in range(cls_ind, self.length, 1000)]

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        item = super().__getitem__(index)
        cls_ind = index % len(MAPPING_FRAME.index)
        cls = MAPPING_FRAME.iloc[cls_ind]
        batch_ind = (index // len(MAPPING_FRAME.index)) % self.typeFactor
        iteration = self.iterations[index // len(MAPPING_FRAME.index) // self.typeFactor]
        item['imagenet_id'] = cls.name
        item['num_idx'] = cls_ind
        item['name'] = cls['name']
        item['iteration'] = iteration
        
        gen_type_dir = self.gen_type if self.gen_type != 'all' else ('div' if batch_ind < 5 else 'clear')
        if gen_type_dir == 'div': item['batch_ind'] = batch_ind
        item['gen_type'] = gen_type_dir
        
        img_name = f'{cls.name}{("_" + str(batch_ind)) if gen_type_dir == "div" else ""}_{str(iteration)}.png'
        img_path = os.path.join(self.dataset_path, gen_type_dir, str(iteration), img_name)
        item['img'] = Image.open(img_path).convert('RGB')
        return item