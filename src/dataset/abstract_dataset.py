from typing import List

import pandas as pd
from torch.utils.data import Dataset


class AbstractDataset(Dataset):
    """
    Dataset with data balancing - getting equal amount of each dataset's items
    The first dataframe represents the main dataset (its values all seen during each epoch)
    """

    def __init__(self, df_list: List[pd.DataFrame], data_dir: str):
        self.data_dir = data_dir
        self.texts = [df['text'].values for df in df_list]
        self.paths = [df['path'].values for df in df_list]
        self.current_idx_list = [0 for _ in df_list]

    def __len__(self):
        return len(self.texts[0]) * len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        pass
