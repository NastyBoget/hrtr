import torch


russian_synthetic_char_set = ' !"%(),-.0123456789:;?АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюя'

russian_char_set = ' !"%(),-./0123456789:;?[]abgnpvx«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё'

russian_kazakh_char_set = ' !(),-.:;?HoАБВГДЕЖЗИЙКЛМНОПРСТУФХЧШЩЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёғҚқҮӨө–—…'

charsets = dict(
    rus=russian_char_set,
    hkr=russian_kazakh_char_set,
    synthetic=russian_synthetic_char_set
)


def check_valid_label(label: str, char_set: str) -> bool:
    return set(label).issubset(char_set)


class ModelOptions:
    
    def __init__(self,
                 saved_model: str = None,
                 Transformation: str = "TPS",
                 FeatureExtraction: str = "ResNet",
                 SequenceModeling: str = "BiLSTM",
                 Prediction: str = "Attn",
                 workers: int = 0,
                 batch_size: int = 192,
                 batch_max_length: int = 25,
                 imgH: int = 32,
                 imgW: int = 100,
                 rgb: bool = False,
                 character: str = russian_char_set,
                 sensitive: bool = True,
                 PAD: bool = False,
                 data_filtering_off: bool = True,
                 num_fiducial: int = 20,
                 output_channel: int = 512,
                 hidden_size: int = 256):
        self.Transformation = Transformation
        self.FeatureExtraction = FeatureExtraction
        self.SequenceModeling = SequenceModeling
        self.Prediction = Prediction
        self.saved_model = saved_model
        self.workers = workers
        self.batch_size = batch_size
        self.batch_max_length = batch_max_length
        self.imgH = imgH
        self.imgW = imgW
        self.rgb = rgb
        if self.rgb:
            self.input_channel = 3
        else:
            self.input_channel = 1
        self.character = character
        if Prediction == "Attn":
            self.num_class = len(self.character) + 2
        else:
            self.num_class = len(self.character) + 1
        self.sensitive = sensitive
        self.PAD = PAD
        self.data_filtering_off = data_filtering_off
        self.num_fiducial = num_fiducial
        self.output_channel = output_channel
        self.hidden_size = hidden_size
        self.num_gpu = torch.cuda.device_count()
