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
        # image = cv2.imread(os.path.join(self.data_dir, self.paths[idx]))
        # image = self.image_preprocessing(image)
        # text = self.texts[idx]
        # encoded = self.label_encoder(text)
        # return {"image": image, "text": text, "encoded": encoded}
        pass
