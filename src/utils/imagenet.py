import os
import pandas as pd
from typing import Union, List

MAPPING_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 
                 '../data/imagenet_class_index.json'))
IMG_NAMES_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 '../data/img_names_by_cat.json'))

MAPPING_FRAME = pd.read_json(MAPPING_PATH, orient='index')
MAPPING_FRAME.rename(columns={0: 'imagenet_id', 1: 'name'}, inplace=True)
IMG_FRAME = pd.read_json(IMG_NAMES_PATH, orient='index')
IMG_FRAME.reindex(MAPPING_FRAME['imagenet_id'])
IMG_FRAME.reset_index(inplace=True, names='imagenet_id')
IMG_FRAME = IMG_FRAME.melt(id_vars=['imagenet_id'], value_name='img_name')
IMG_FRAME = IMG_FRAME.loc[~IMG_FRAME['img_name'].isnull()]
IMG_FRAME.drop('variable', axis=1, inplace=True)
MAPPING_FRAME.reset_index(names='num_idx', inplace=True)
MAPPING_FRAME.set_index('imagenet_id', inplace=True)


def get_categories(max_nr: Union[int,None] = None, seed: Union[int,None] = None) -> List[str]:
    """Get all or a subset of categories represented in the dataset

    Args:
        max_nr (int, optional): The maximum number classes to return. Defaults to None.
        seed (int, optional): The seed for the random subset. Defaults to None.

    Returns:
        List[str]: All/a random subset of classes of the dataset
    """
    return MAPPING_FRAME.index.tolist() if max_nr is None else \
        MAPPING_FRAME.sample(max_nr, random_state=seed, replace=False).index.tolist()
    
def get_index_for_imagenet_id(imagenet_id: str) -> int:
    """Get the numerical index for a given imagenet id

    Args:
        imagenet_id (str): the imagenet id

    Returns:
        int: the number of the imagenet id
    """
    return MAPPING_FRAME.loc[imagenet_id]['num_idx']

def get_imagenet_id_for_names(names: List[str]) -> List[str]:
    """Return the imagenet ids for a list of readable class names

    Args:
        names (List[str]): The list of readable names

    Returns:
        List[str]: The list of imagenet ids
    """
    return [MAPPING_FRAME[MAPPING_FRAME['name']==name].index[0] for name in names]

def get_names_for_imagenet_ids(imagenet_ids: List[str]) -> List[str]:
    """Return the readable names for a list of imagenet ids

    Args:
        imagenet_ids (List[str]): the list of imagenet ids i.e. n07753275

    Returns:
        List[str]: The list of readable names i.e. pineapple
    """
    return MAPPING_FRAME.loc[imagenet_ids]['name'].tolist()