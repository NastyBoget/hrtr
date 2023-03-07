import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TPSSpatialTransformerNetwork(nn.Module):
    """ Rectification Network of RARE, namely TPS based STN """

    def __init__(self, f, i_size, i_r_size, i_channel_num=1):
        """ Based on RARE TPS
        :param f:
        :param i_size: (height, width) of the input image I
        :param i_r_size : (height, width) of the rectified image I_r
        :param i_channel_num : the number of channels of the input image I
        :return: rectified image [batch_size x I_channel_num x I_r_height x I_r_width]
        """
        super(TPSSpatialTransformerNetwork, self).__init__()
        self.f = f
        self.i_size = i_size
        self.i_r_size = i_r_size  # = (I_r_height, I_r_width)
        self.i_channel_num = i_channel_num
        self.localization_network = LocalizationNetwork(self.f, self.i_channel_num)
        self.grid_generator = GridGenerator(self.f, self.i_r_size)

    def forward(self, batch_i):
        """
        :param batch_i: Batch Input Image [batch_size x I_channel_num x I_height x I_width]
        """
        batch_c_prime = self.localization_network(batch_i)  # batch_size x K x 2
        build_p_prime = self.grid_generator.build_p_prime(batch_c_prime)  # batch_size x n (= I_r_width x I_r_height) x 2
        build_p_prime_reshape = build_p_prime.reshape([build_p_prime.size(0), self.i_r_size[0], self.i_r_size[1], 2])
        
        if torch.__version__ > "1.2.0":
            batch_i_r = F.grid_sample(batch_i, build_p_prime_reshape, padding_mode='border', align_corners=True)
        else:
            batch_i_r = F.grid_sample(batch_i, build_p_prime_reshape, padding_mode='border')

        return batch_i_r


class LocalizationNetwork(nn.Module):
    """ Localization Network of RARE, which predicts C' (K x 2) from I (I_width x I_height) """

    def __init__(self, f, i_channel_num):
        super(LocalizationNetwork, self).__init__()
        self.f = f
        self.i_channel_num = i_channel_num
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.i_channel_num, out_channels=64, kernel_size=3, stride=1, padding=1,
                      bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 64 x I_height/2 x I_width/2
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 128 x I_height/4 x I_width/4
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 256 x I_height/8 x I_width/8
            nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)  # batch_size x 512
        )

        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, self.f * 2)

        # Init fc2 in LocalizationNetwork
        self.localization_fc2.weight.data.fill_(0)
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(f / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(f / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(f / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)

    def forward(self, batch_i):
        """
        :param batch_i: Batch Input Image [batch_size x I_channel_num x I_height x I_width]
        :return: Predicted coordinates of fiducial points for input batch [batch_size x F x 2]
        """
        batch_size = batch_i.size(0)
        features = self.conv(batch_i).view(batch_size, -1)
        batch_c_prime = self.localization_fc2(self.localization_fc1(features)).view(batch_size, self.f, 2)
        return batch_c_prime


class GridGenerator(nn.Module):
    """ Grid Generator of RARE, which produces P_prime by multipling T with P """

    def __init__(self, f, i_r_size):
        """ Generate p_hat and inv_delta_c for later """
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.i_r_height, self.i_r_width = i_r_size
        self.f = f
        self.c = self._build_c(self.f)  # F x 2
        self.p = self._build_p(self.i_r_width, self.i_r_height)
        # for multi-gpu, you need register buffer
        self.register_buffer("inv_delta_c", torch.tensor(self._build_inv_delta_c(self.f, self.c)).float())  # F+3 x F+3
        self.register_buffer("p_hat", torch.tensor(self._build_p_hat(self.f, self.c, self.p)).float())  # n x F+3

    def _build_c(self, F):
        """ Return coordinates of fiducial points in I_r; C """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # F x 2

    def _build_inv_delta_c(self, f, c):
        """ Return inv_delta_c which is needed to calculate T """
        hat_c = np.zeros((f, f), dtype=float)  # F x F
        for i in range(0, f):
            for j in range(i, f):
                r = np.linalg.norm(c[i] - c[j])
                hat_c[i, j] = r
                hat_c[j, i] = r
        np.fill_diagonal(hat_c, 1)
        hat_c = (hat_c ** 2) * np.log(hat_c)
        # print(C.shape, hat_c.shape)
        delta_c = np.concatenate(  # F+3 x F+3
            [
                np.concatenate([np.ones((f, 1)), c, hat_c], axis=1),  # F x F+3
                np.concatenate([np.zeros((2, 3)), np.transpose(c)], axis=1),  # 2 x F+3
                np.concatenate([np.zeros((1, 3)), np.ones((1, f))], axis=1)  # 1 x F+3
            ],
            axis=0
        )
        inv_delta_c = np.linalg.inv(delta_c)
        return inv_delta_c  # F+3 x F+3

    def _build_p(self, i_r_width, i_r_height):
        i_r_grid_x = (np.arange(-i_r_width, i_r_width, 2) + 1.0) / i_r_width  # self.I_r_width
        i_r_grid_y = (np.arange(-i_r_height, i_r_height, 2) + 1.0) / i_r_height  # self.I_r_height
        p = np.stack(  # self.I_r_width x self.I_r_height x 2
            np.meshgrid(i_r_grid_x, i_r_grid_y),
            axis=2
        )
        return p.reshape([-1, 2])  # n (= self.I_r_width x self.I_r_height) x 2

    def _build_p_hat(self, f, c, p):
        n = p.shape[0]  # n (= self.I_r_width x self.I_r_height)
        p_tile = np.tile(np.expand_dims(p, axis=1), (1, f, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2
        c_tile = np.expand_dims(c, axis=0)  # 1 x F x 2
        p_diff = p_tile - c_tile  # n x F x 2
        rbf_norm = np.linalg.norm(p_diff, ord=2, axis=2, keepdims=False)  # n x F
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
        p_hat = np.concatenate([np.ones((n, 1)), p, rbf], axis=1)
        return p_hat  # n x F+3

    def build_p_prime(self, batch_c_prime):
        """ Generate Grid from batch_C_prime [batch_size x F x 2] """
        batch_size = batch_c_prime.size(0)
        batch_inv_delta_c = self.inv_delta_c.repeat(batch_size, 1, 1)
        batch_p_hat = self.p_hat.repeat(batch_size, 1, 1)
        batch_c_prime_with_zeros = torch.cat((batch_c_prime, torch.zeros(batch_size, 3, 2).float().to(device)), dim=1)  # batch_size x F+3 x 2
        batch_t = torch.bmm(batch_inv_delta_c, batch_c_prime_with_zeros)  # batch_size x F+3 x 2
        batch_p_prime = torch.bmm(batch_p_hat, batch_t)  # batch_size x n x 2
        return batch_p_prime  # batch_size x n x 2
