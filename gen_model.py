import torch
import torch.nn as nn
import torch.nn.init as init


class ZipLayer(nn.Module):
    """Zip Module"""

    def __init__(
        self,
        in_channels,
        squeeze_channels,
        e1x1_channels,
        e3x3_channels
    ):
        super(ZipLayer, self).__init__()

        self.in_channels = in_channels
        self.squeeze_channels = squeeze_channels
        self.e1x1_channels = e1x1_channels
        self.e3x3_channels = e3x3_channels

        self.squeeze_layer = self.get_squeeze_layer()
        self.expand_1x1_layer = self.get_expand_1x1_layer()
        self.expand_3x3_layer = self.get_expand_3x3_layer()

    def get_squeeze_layer(self):
        layers = []

        layers.append(nn.Conv2d(self.in_channels,
                                self.squeeze_channels,
                                kernel_size=1))
        layers.append(nn.BatchNorm2d(num_features=self.squeeze_channels))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def get_expand_1x1_layer(self):
        layers = []

        layers.append(nn.Conv2d(self.squeeze_channels,
                                self.e1x1_channels,
                                kernel_size=1))
        layers.append(nn.BatchNorm2d(num_features=self.e1x1_channels))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def get_expand_3x3_layer(self):
        layers = []

        layers.append(nn.Conv2d(self.squeeze_channels,
                                self.e3x3_channels,
                                kernel_size=3,
                                padding=1))
        layers.append(nn.BatchNorm2d(num_features=self.e3x3_channels))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.squeeze_layer(x)
        return torch.cat([
            self.expand_1x1_layer(y),
            self.expand_3x3_layer(y)
        ], 1)


class ZipBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 num_layers,
                 compress_factor,
                 expand_factor,
                 expand_interval):
        super(ZipBlock, self).__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.compress_factor = compress_factor
        self.expand_factor = expand_factor
        self.expand_interval = expand_interval

        self.net = self.get_network()

    def get_network(self):
        layers = []

        in_channels = self.in_channels
        for i in range(self.num_layers):
            squeeze_channels = in_channels // self.compress_factor

            out_channels = in_channels
            if (i + 1) % self.expand_interval == 0:
                out_channels *= self.expand_factor

            layers.append(ZipLayer(in_channels=in_channels,
                                   squeeze_channels=squeeze_channels,
                                   e1x1_channels=out_channels // 2,
                                   e3x3_channels=out_channels // 2))

            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ZipNet(nn.Module):

    """ZFNetLite"""

    def __init__(
        self,
        channels,
        class_count
    ):
        super(ZipNet, self).__init__()
        self.channels = channels
        self.class_count = class_count

        self.conv_net = self.get_conv_net()
        self.fc_net = self.get_fc_net()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)

                else:
                    init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def get_conv_net(self):
        layers = []

        # in_channels = self.channels, out_channels = 64
        # kernel_size = 3x3, stride = 2
        layers.append(nn.Conv2d(self.channels, 64, kernel_size=3, stride=2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        layers.append(nn.BatchNorm2d(num_features=64))

        # in_channels = 64, squeeze_channels = 16
        # e1x1_channels = 32, e3x3_channels = 32 -> out_channels = 64
        # layers.append(ZipModule(64, 16, 32, 32))

        # in_channels = 64, squeeze_channels = 16
        # e1x1_channels = 64, e3x3_channels = 64 -> out_channels = 128
        # layers.append(ZipModule(64, 16, 64, 64))

        # in_channels = 128, squeeze_channels = 16
        # e1x1_channels = 64, e3x3_channels = 64 -> out_channels = 128
        # layers.append(ZipModule(128, 16, 64, 64))

        # in_channels = 128, squeeze_channels = 16
        # e1x1_channels = 64, e3x3_channels = 64 -> out_channels = 128
        # layers.append(ZipModule(128, 16, 64, 64))
        layers.append(ZipBlock(in_channels=64,
                               num_layers=4,
                               compress_factor=4,
                               expand_factor=2,
                               expand_interval=2))

        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        # in_channels = 128, squeeze_channels = 32
        # e1x1_channels = 128, e3x3_channels = 128 -> out_channels = 256
        # layers.append(ZipModule(128, 32, 128, 128))

        # in_channels = 256, squeeze_channels = 32
        # e1x1_channels = 128, e3x3_channels = 128 -> out_channels = 256
        # layers.append(ZipModule(256, 32, 128, 128))
        layers.append(ZipBlock(in_channels=256,
                               num_layers=2,
                               compress_factor=8,
                               expand_factor=1,
                               expand_interval=1))

        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        # in_channels = 256, squeeze_channels = 48
        # e1x1_channels = 192, e3x3_channels = 192 -> out_channels = 384
        # layers.append(ZipModule(256, 48, 192, 192))

        # in_channels = 384, squeeze_channels = 48
        # e1x1_channels = 192, e3x3_channels = 192 -> out_channels = 384
        # layers.append(ZipModule(384, 48, 192, 192))

        # in_channels = 384, squeeze_channels = 64
        # e1x1_channels = 256, e3x3_channels = 256 -> out_channels = 512
        # layers.append(ZipModule(384, 64, 256, 256))

        # in_channels = 512, squeeze_channels = 64
        # e1x1_channels = 256, e3x3_channels = 256 -> out_channels = 512
        # layers.append(ZipModule(512, 64, 256, 256))
        layers.append(ZipBlock(in_channels=256,
                               num_layers=4,
                               compress_factor=8,
                               expand_factor=2,
                               expand_interval=2))
        return nn.Sequential(*layers)

    def get_fc_net(self):
        layers = []

        self.final_conv = nn.Conv2d(1024, self.class_count, kernel_size=1)

        layers.append(nn.Dropout())
        layers.append(self.final_conv)
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.AvgPool2d(13, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv_net(x)
        y = self.fc_net(y)
        return y.view(y.size(0), self.class_count)
