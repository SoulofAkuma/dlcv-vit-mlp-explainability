import os
from PIL import Image
import pandas as pd
from src.utils.IndexDataset import IndexDataset
from typing import Union, List
from src.utils.imagenet import MAPPING_FRAME, IMG_FRAME

class ImageNetDataset(IndexDataset):

    def __init__(self, dataset_path, partition='val'):
        super().__init__()
        self.data_path = os.path.join(dataset_path, partition)

    def __len__(self):
        return len(IMG_FRAME.index)

    def get_images_from_imgnet_id(self, imagenet_id: str) -> List:
        """Get all the items in the dataset from the given class

        Args:
            imagenet_id (str): The given imagenet class i.e. n0000000

        Returns:
            List: The list of items from the dataset
        """
        indices = IMG_FRAME.index[IMG_FRAME['imagenet_id'] == imagenet_id].tolist()
        return [self[index] for index in indices]    

    def __getitem__(self, index):
        item = super().__getitem__(index)
        img_data = IMG_FRAME.iloc[index]
        cat_data = MAPPING_FRAME.loc[img_data['imagenet_id']]
        item['imagenet_id'] = img_data['imagenet_id']
        item['num_idx'] = cat_data['num_idx']
        item['name'] = cat_data['name']
        item['img'] = Image.open(
            os.path.join(self.data_path, img_data['imagenet_id'], img_data['img_name'])) \
            .convert('RGB')
        return item