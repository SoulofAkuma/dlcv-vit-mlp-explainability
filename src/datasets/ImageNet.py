import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from typing import Union, List

MAPPING_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 
                 '../../data/imagenet_class_index.json'))
IMG_NAMES = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 '../../data/img_names_by_cat.json'))

MAPPING_FRAME = pd.read_json(MAPPING_PATH, orient='index')
MAPPING_FRAME.rename(columns={0: 'imagenet_id', 1: 'category'}, inplace=True)
IMG_FRAME = pd.read_json(IMG_NAMES, orient='index')
IMG_FRAME.reindex(MAPPING_FRAME['imagenet_id'])
IMG_FRAME.reset_index(inplace=True, names='imagenet_id')
IMG_FRAME = IMG_FRAME.melt(id_vars=['imagenet_id'], value_name='img_name')
IMG_FRAME = IMG_FRAME.loc[~IMG_FRAME['img_name'].isnull()]
IMG_FRAME.drop('variable', axis=1, inplace=True)
MAPPING_FRAME.reset_index(names='class', inplace=True)
MAPPING_FRAME.set_index('imagenet_id', inplace=True)

class ImageNetDataset(Dataset):

    def __init__(self, dataset_path, partition='val'):
        self.data_path = os.path.join(dataset_path, partition)

    def __len__(self):
        return len(IMG_FRAME.index)
    
    def get_categories(self, max_nr: Union[int,None] = None, seed: Union[int,None] = None) -> List[str]:
        """Get all or a subset of categories represented in the dataset

        Args:
            max_nr (int, optional): The maximum number classes to return. Defaults to None.
            seed (int, optional): The seed for the random subset. Defaults to None.

        Returns:
            List[str]: All/a random subset of classes of the dataset
        """
        return MAPPING_FRAME.index.tolist() if max_nr is None else \
            MAPPING_FRAME.sample(max_nr, random_state=seed, replace=False).index.tolist()

    def get_images_from_class(self, imagenet_id: str) -> List:
        """Get all the items in the dataset from the given class

        Args:
            imagenet_id (str): The given imagenet class i.e. n0000000

        Returns:
            List: The list of items from the dataset
        """
        indices = IMG_FRAME.index[IMG_FRAME['imagenet_id'] == imagenet_id].tolist()
        return [self[index] for index in indices]    

    def __getitem__(self, index):
        item = {}
        img_data = IMG_FRAME.iloc[index]
        cat_data = MAPPING_FRAME.loc[img_data['imagenet_id']]
        item['imagenet_id'] = img_data['imagenet_id']
        item['class'] = cat_data['class']
        item['category'] = cat_data['category']
        item['img'] = Image.open(
            os.path.join(self.data_path, img_data['imagenet_id'], img_data['img_name'])) \
            .convert('RGB')
        return item