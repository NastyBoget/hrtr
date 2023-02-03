"""
A modified version of AttentionHTR repository https://github.com/dmitrijsk/AttentionHTR
License: https://github.com/dmitrijsk/AttentionHTR/blob/main/model/LICENSE.md
"""

import torch.nn as nn

from src.model.feature_extraction.rcnn import RCNN_FeatureExtractor
from src.model.feature_extraction.resnet import ResNet_FeatureExtractor
from src.model.feature_extraction.vgg import VGG_FeatureExtractor
from src.model.prediction.attention import Attention
from src.model.preprocessing.transformation import TPS_SpatialTransformerNetwork
from src.model.sequence_modeling.bilstm import BidirectionalLSTM


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.img_h, opt.img_w), I_r_size=(opt.img_h, opt.img_w), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(img_h/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (img_h/16-1) -> 1

        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, inp, text, is_train=True):
        if not self.stages['Trans'] == "None":
            inp = self.Transformation(inp)

        visual_feature = self.FeatureExtraction(inp)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction

    def reset_output(self, charset: str) -> None:
        self.opt.character = charset
        if self.opt.Prediction == 'CTC':
            self.opt.num_class = len(charset) + 1
        else:
            self.opt.num_class = len(charset) + 2

        if self.opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, self.opt.num_class)
        elif self.opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, self.opt.hidden_size, self.opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')
