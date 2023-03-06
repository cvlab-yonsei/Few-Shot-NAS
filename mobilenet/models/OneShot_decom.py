import math
import torch.nn as nn
from models.layers import OPS


class SuperNet_decom(nn.Module):
    def __init__(self, K, search_space, affine, track_running_stats, n_class=1000, input_size=224, width_mult=1.):
        super(SuperNet_decom, self).__init__()

#        self.interverted_residual_setting = [
#            # channel, layers, stride
#            [32//K,  4, 2],
#            [56//K,  4, 2],
#            [112//K, 4, 2],
#            [128//K, 4, 1],
#            [256//K, 4, 2],
#            [432//K, 1, 1],
#        ]
#
#        input_channel = int((40//K) * width_mult)
#        first_cell_width = int((24//K) * width_mult)

        self.interverted_residual_setting = [
            # channel, layers, stride
            [24//K,  4, 2],
            [40//K,  4, 2],
            [80//K,  4, 2],
            [96//K,  4, 1],
            [192//K, 4, 2],
            [320//K, 1, 1],
        ]

        input_channel    = int((32//K) * width_mult)
        first_cell_width = int((16//K) * width_mult)

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
                if i == 0:
                    op_list = [ nn.ModuleList([OPS[op_name](input_channel, output_channel, s, affine, track_running_stats) for op_name in search_space[:-1]]) for _ in range(K) ]
                else:
                    op_list = [ nn.ModuleList([OPS[op_name](input_channel, output_channel, 1, affine, track_running_stats) for op_name in search_space]) for _ in range(K) ]
                op_list = nn.ModuleList(op_list)
                self.blocks.append( op_list )
                self.choices.append( len(op_list) )
                input_channel = output_channel

        # last_channel = int((1728//K) * width_mult)
        last_channel = int((1280//K) * width_mult)
        self.feature_mix_layer = nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channel, affine=affine, track_running_stats=track_running_stats),
            nn.ReLU6(inplace=True)
        ) 
        self.avgpool    = nn.AvgPool2d(input_size//32)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, n_class),
        )

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

    def get_flops(self, arch):
        def count_conv_flop(layer, x):
            out_h = int(x / layer.stride[0])
            out_w = int(x / layer.stride[1])
            delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * out_h * out_w / layer.groups
            return delta_ops

    
        flops = count_conv_flop(self.first_conv[0], 224)
        flops += count_conv_flop(self.first_block.depth_conv[0], 112)
        flops += count_conv_flop(self.first_block.point_linear[0], 112)

        sizes = [112] + [56]*4 + [28]*4 + [14]*8 + [7]*4
        for ops, op_ind, ss in zip(self.blocks, arch, sizes):
            if op_ind != 6:
                flops += count_conv_flop(ops[op_ind].inverted_bottleneck[0], ss)
                flops += count_conv_flop(ops[op_ind].depth_conv[0], ss)
                flops += count_conv_flop(ops[op_ind].point_linear[0], ss)

        flops += count_conv_flop(self.feature_mix_layer[0], sizes[-1])
        flops += self.classifier[0].weight.numel()
        return flops

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