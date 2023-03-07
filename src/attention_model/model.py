"""
A modified version of AttentionHTR repository https://github.com/dmitrijsk/AttentionHTR
License: https://github.com/dmitrijsk/AttentionHTR/blob/main/model/LICENSE.md
"""
import logging

import torch
import torch.nn as nn

from attention import Attention
from bilstm import BidirectionalLSTM
from resnet import ResNetFeatureExtractor
from transformation import TPSSpatialTransformerNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module):

    def __init__(self, opt, logger: logging.Logger):
        super(Model, self).__init__()
        self.opt = opt
        self.logger = logger
        self.stages = {'Trans': opt.transformation, 'Feat': opt.feature_extraction, 'Seq': opt.sequence_modeling, 'Pred': opt.prediction}

        self.transformation = TPSSpatialTransformerNetwork(
            f=opt.num_fiducial, i_size=(opt.img_h, opt.img_w), i_r_size=(opt.img_h, opt.img_w), i_channel_num=opt.input_channel
        )
        self.feature_extraction = ResNetFeatureExtractor(opt.input_channel, opt.output_channel)
        self.feature_extraction_output = opt.output_channel  # int(img_h/16-1) * 512
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (img_h/16-1) -> 1

        self.sequence_modeling = nn.Sequential(
            BidirectionalLSTM(self.feature_extraction_output, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size)
        )
        self.sequence_modeling_output = opt.hidden_size
        self.prediction = Attention(self.sequence_modeling_output, opt.hidden_size, opt.num_class)

    def forward(self, inp, text, is_train=True):
        inp = self.transformation(inp)
        visual_feature = self.feature_extraction(inp)
        visual_feature = self.adaptive_avg_pool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)
        contextual_feature = self.sequence_modeling(visual_feature)
        prediction = self.prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)
        return prediction
