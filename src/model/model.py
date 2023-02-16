"""
A modified version of AttentionHTR repository https://github.com/dmitrijsk/AttentionHTR
License: https://github.com/dmitrijsk/AttentionHTR/blob/main/model/LICENSE.md
"""
import logging

import torch.nn as nn

from src.model.feature_extraction.rcnn import RCNNFeatureExtractor
from src.model.feature_extraction.resnet import ResNetFeatureExtractor
from src.model.feature_extraction.vgg import VGGFeatureExtractor
from src.model.prediction.attention import Attention
from src.model.preprocessing.transformation import TPSSpatialTransformerNetwork
from src.model.sequence_modeling.bigru import BidirectionalGRU
from src.model.sequence_modeling.bilstm import BidirectionalLSTM


class Model(nn.Module):

    def __init__(self, opt, logger: logging.Logger):
        super(Model, self).__init__()
        self.opt = opt
        self.logger = logger
        self.stages = {'Trans': opt.transformation, 'Feat': opt.feature_extraction, 'Seq': opt.sequence_modeling, 'Pred': opt.prediction}

        if opt.transformation == 'TPS':
            self.transformation = TPSSpatialTransformerNetwork(
                f=opt.num_fiducial, i_size=(opt.img_h, opt.img_w), i_r_size=(opt.img_h, opt.img_w), i_channel_num=opt.input_channel
            )
        else:
            self.logger.info('No Transformation module specified')
            self.stages['Trans'] = "None"

        if opt.feature_extraction == 'VGG':
            self.feature_extraction = VGGFeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.feature_extraction == 'RCNN':
            self.feature_extraction = RCNNFeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.feature_extraction == 'ResNet':
            self.feature_extraction = ResNetFeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.feature_extraction_output = opt.output_channel  # int(img_h/16-1) * 512
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (img_h/16-1) -> 1

        if opt.sequence_modeling == 'BiLSTM':
            self.sequence_modeling = nn.Sequential(
                BidirectionalLSTM(self.feature_extraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size)
            )
            self.sequence_modeling_output = opt.hidden_size
        elif opt.sequence_modeling == 'BiGRU':
            self.sequence_modeling = nn.Sequential(
                BidirectionalGRU(self.feature_extraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalGRU(opt.hidden_size, opt.hidden_size, opt.hidden_size)
            )
            self.sequence_modeling_output = opt.hidden_size
        else:
            self.logger.info('No SequenceModeling module specified')
            self.stages['Seq'] = "None"
            self.sequence_modeling_output = self.feature_extraction_output

        if opt.prediction == 'CTC':
            self.prediction = nn.Linear(self.sequence_modeling_output, opt.num_class)
        elif opt.prediction == 'Attn':
            self.prediction = Attention(self.sequence_modeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, inp, text, is_train=True):
        if not self.stages['Trans'] == "None":
            inp = self.transformation(inp)

        visual_feature = self.feature_extraction(inp)
        visual_feature = self.adaptive_avg_pool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        if not self.stages['Seq'] == 'None':
            contextual_feature = self.sequence_modeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        if self.stages['Pred'] == 'CTC':
            prediction = self.prediction(contextual_feature.contiguous())
        else:
            prediction = self.prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction

    def reset_output(self, charset: str) -> None:
        self.opt.character = charset
        self.opt.character = self.opt.character if self.opt.sensitive else "".join(list(set(self.opt.character.lower())))
        self.opt.num_class = len(self.opt.character) + 1 if self.opt.prediction == 'CTC' else len(self.opt.character) + 2

        if self.opt.prediction == 'CTC':
            self.prediction = nn.Linear(self.sequence_modeling_output, self.opt.num_class)
        elif self.opt.prediction == 'Attn':
            self.prediction = Attention(self.sequence_modeling_output, self.opt.hidden_size, self.opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')
