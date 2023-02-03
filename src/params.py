class ModelOptions:
    
    def __init__(self,
                 saved_model: str = None,
                 transformation: str = "TPS",
                 feature_extraction: str = "ResNet",
                 sequence_modeling: str = "BiLSTM",
                 prediction: str = "Attn",
                 batch_max_length: int = 40,
                 img_h: int = 32,
                 img_w: int = 100,
                 rgb: bool = False,
                 character: str = "",
                 sensitive: bool = True,
                 pad: bool = False,
                 num_fiducial: int = 20,
                 output_channel: int = 512,
                 hidden_size: int = 256):
        self.transformation = transformation
        self.feature_extraction = feature_extraction
        self.sequence_modeling = sequence_modeling
        self.prediction = prediction
        self.saved_model = saved_model
        self.batch_max_length = batch_max_length
        self.img_h = img_h
        self.img_w = img_w
        self.rgb = rgb
        self.input_channel = 3 if self.rgb else 1
        self.character = character
        self.num_class = len(self.character) + 2 if prediction == "Attn" else len(self.character) + 1
        self.sensitive = sensitive
        self.pad = pad
        self.num_fiducial = num_fiducial
        self.output_channel = output_channel
        self.hidden_size = hidden_size
