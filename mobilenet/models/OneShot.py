import math
import torch.nn as nn
from models.layers import FrozenBatchNorm2d, Select_one_OP, OPS 


class SuperNet(nn.Module):
    def __init__(self, search_space, affine, track_running_stats, freeze_bn, n_class=1000, input_size=224, width_mult=1.):
        super(SuperNet, self).__init__()

        self.interverted_residual_setting = [
            # channel, layers, stride
            [32,  4, 2],
            [56,  4, 2],
            [112, 4, 2],
            [128, 4, 1],
            [256, 4, 2],
            [432, 1, 1],
        ]

        input_channel    = int(40 * width_mult)
        first_cell_width = int(24 * width_mult)

        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel, affine=affine, track_running_stats=track_running_stats),
            nn.ReLU6(inplace=True),
        )

        self.first_block = OPS['3x3_MBConv1'](input_channel, first_cell_width, 1, affine, track_running_stats)
        input_channel = first_cell_width

        self.blocks  = nn.ModuleList()
        self.choices = []
        for c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                self.choices.append( list(range(len(search_space))) )
                if i == 0:
                    self.blocks.append( Select_one_OP(search_space, input_channel, output_channel, s, affine, track_running_stats) )
                else:
                    self.blocks.append( Select_one_OP(search_space, input_channel, output_channel, 1, affine, track_running_stats) )
                    self.choices[-1].append(-1) # Add an identity layer
                input_channel = output_channel

        last_channel = int(1728 * width_mult)
        self.feature_mix_layer = nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channel, affine=affine, track_running_stats=track_running_stats),
            nn.ReLU6(inplace=True)
        ) 
        self.avgpool    = nn.AvgPool2d(input_size//32)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, n_class),
        )

        if freeze_bn: self.freeze_bn()
        self.initialize()

    def initialize(self): 
        for m in self.modules():
            if isinstance(m, nn.Conv2d): # he_fout
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # if init_div_groups:
                #     n /= m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()

    def freeze_bn(self):
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)

    def arch_uniform_sampling(self):
        rand_arch = []
        for i, ops in enumerate(operations):
            k = np.random.randint(len(ops))
            select_op = ops[k]
            rand_arch.append(select_op)

        return rand_arch

    def forward(self, x, arch):
        x = self.first_conv(x)
        x = self.first_block(x)

        for ops, op_ind in zip(self.blocks, arch):
            x = ops[op_ind](x)
        x = self.feature_mix_layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
