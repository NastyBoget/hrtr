import cv2
from torch.utils.data import Dataset


class HTRDataset(Dataset):

    def __init__(self, df, config, label_encoder, image_preprocessing, transforms=None):
        self.config = config
        self.texts = df['text'].values
        self.paths = df['path'].values
        self.label_encoder = label_encoder
        self.image_preprocessing = image_preprocessing
        self.transforms = transforms

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        image = cv2.imread(f'{self.config.data_dir}/{self.paths[idx]}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.image_preprocessing(image)
        text = self.texts[idx]
        encoded = self.label_encoder(text)
        return {"image": image, "text": text, "encoded": encoded}
