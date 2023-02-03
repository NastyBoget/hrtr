import torch.nn as nn
import torch.nn.functional as F


class RCNNFeatureExtractor(nn.Module):
    """
    FeatureExtractor of GRCNN
    (https://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf)
    """

    def __init__(self, input_channel, output_channel=512):
        super(RCNNFeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64 x 16 x 50
            GRCL(self.output_channel[0], self.output_channel[0], num_iteration=5, kernel_size=3, pad=1),
            nn.MaxPool2d(2, 2),  # 64 x 8 x 25
            GRCL(self.output_channel[0], self.output_channel[1], num_iteration=5, kernel_size=3, pad=1),
            nn.MaxPool2d(2, (2, 1), (0, 1)),  # 128 x 4 x 26
            GRCL(self.output_channel[1], self.output_channel[2], num_iteration=5, kernel_size=3, pad=1),
            nn.MaxPool2d(2, (2, 1), (0, 1)),  # 256 x 2 x 27
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 2, 1, 0, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True))  # 512 x 1 x 26

    def forward(self, inp):
        return self.conv_net(inp)


# For Gated RCNN
class GRCL(nn.Module):

    def __init__(self, input_channel, output_channel, num_iteration, kernel_size, pad):
        super(GRCL, self).__init__()
        self.wgf_u = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False)
        self.wgr_x = nn.Conv2d(output_channel, output_channel, 1, 1, 0, bias=False)
        self.wf_u = nn.Conv2d(input_channel, output_channel, kernel_size, 1, pad, bias=False)
        self.wr_x = nn.Conv2d(output_channel, output_channel, kernel_size, 1, pad, bias=False)

        self.bn_x_init = nn.BatchNorm2d(output_channel)

        self.num_iteration = num_iteration
        self.grcl = [GRCLUnit(output_channel) for _ in range(num_iteration)]
        self.grcl = nn.Sequential(*self.grcl)

    def forward(self, input):
        """
        The input of GRCL is consistant over time t, which is denoted by u(0)
        thus wgf_u / wf_u is also consistant over time t.
        """
        wgf_u = self.wgf_u(input)
        wf_u = self.wf_u(input)
        x = F.relu(self.bn_x_init(wf_u))

        for i in range(self.num_iteration):
            x = self.grcl[i](wgf_u, self.wgr_x(x), wf_u, self.wr_x(x))

        return x


class GRCLUnit(nn.Module):

    def __init__(self, output_channel):
        super(GRCLUnit, self).__init__()
        self.bn_gfu = nn.BatchNorm2d(output_channel)
        self.bn_grx = nn.BatchNorm2d(output_channel)
        self.bn_fu = nn.BatchNorm2d(output_channel)
        self.bn_rx = nn.BatchNorm2d(output_channel)
        self.bn_gx = nn.BatchNorm2d(output_channel)

    def forward(self, wgf_u, wgr_x, wf_u, wr_x):
        g_first_term = self.bn_gfu(wgf_u)
        g_second_term = self.bn_grx(wgr_x)
        g = F.sigmoid(g_first_term + g_second_term)

        x_first_term = self.bn_fu(wf_u)
        x_second_term = self.bn_gx(self.bn_rx(wr_x) * g)
        x = F.relu(x_first_term + x_second_term)

        return x
