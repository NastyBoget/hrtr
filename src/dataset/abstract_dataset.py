import pandas as pd
from torch.utils.data import Dataset


class AbstractDataset(Dataset):

    def __init__(self, df: pd.DataFrame, data_dir: str):
        self.data_dir = data_dir
        self.texts = df['text'].values
        self.paths = df['path'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        pass
