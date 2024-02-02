from torch.utils.data.dataset import Dataset

class IndexDataset(Dataset):
    """A simplistic class to include the index in the 

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        return {'dataset_index': index}