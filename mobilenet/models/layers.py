import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['OPS', 'SearchSpaceNames']


OPS = {
    '3x3_MBConv1': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual(C_in, C_out, 3, 1, stride, 1, affine, track_running_stats),

    '3x3_MBConv3': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual(C_in, C_out, 3, 1, stride, 3, affine, track_running_stats),
    '3x3_MBConv6': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual(C_in, C_out, 3, 1, stride, 6, affine, track_running_stats),
    '5x5_MBConv3': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual(C_in, C_out, 5, 2, stride, 3, affine, track_running_stats),
    '5x5_MBConv6': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual(C_in, C_out, 5, 2, stride, 6, affine, track_running_stats),
    '7x7_MBConv3': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual(C_in, C_out, 7, 3, stride, 3, affine, track_running_stats),
    '7x7_MBConv6': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual(C_in, C_out, 7, 3, stride, 6, affine, track_running_stats),
}


PROXYLESS_SPACE = [
    '3x3_MBConv3', '3x3_MBConv6', 
    '5x5_MBConv3', '5x5_MBConv6', 
    '7x7_MBConv3', '7x7_MBConv6'
]


SearchSpaceNames = {
    "proxyless": PROXYLESS_SPACE, 
}


class Select_one_OP(nn.Module):
    def __init__(self, search_space, C_in, C_out, stride, affine, track_running_stats):
        super(Select_one_OP, self).__init__()

        self._ops = nn.ModuleList()
        for op_name in search_space:
            self._ops.append( OPS[op_name](C_in, C_out, stride, affine, track_running_stats) )

        if stride == 1:
            self._ops += [ Identity() ]

    def forward(self, x, ind):
        return self._ops[ind](x)
  

class InvertedResidual(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, padding, stride, expand_ratio, affine, track_running_stats):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio

        self.use_res_connect = self.stride == 1 and C_in == C_out

        hidden_dim = round(C_in * expand_ratio)

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(
                nn.Conv2d(C_in, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim, affine=affine, track_running_stats=track_running_stats),
                nn.ReLU6(inplace=True),
            )

        self.depth_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim, affine=affine, track_running_stats=track_running_stats),
            nn.ReLU6(inplace=True),
        )

        self.point_linear = nn.Sequential(
            nn.Conv2d(hidden_dim, C_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
        )

    def forward(self, x):
        inputs = x

        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x) 
        x = self.depth_conv(x)
        x = self.point_linear(x)

        if self.use_res_connect:
            return inputs + x
        else:
            return x


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.
    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.
    Other pre-trained backbone models may contain all 4 parameters.
    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    _version = 3
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long)) # NOTE: added due to unexpected parameters when loading pre    -trained weights

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype  # may be half
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        version = None # NOTE: added

        if version is None or version < 2:
            # No running_mean/var in early versions
            # This will silent the warnings
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        # NOTE: if a checkpoint is trained with BatchNorm and loaded (together with
        # version number) to FrozenBatchNorm, running_var will be wrong. One solution
        # is to remove the version number from the checkpoint.
        if version is not None and version < 3:
            logger = logging.getLogger(__name__)
            logger.info("FrozenBatchNorm {} is upgraded to version 3.".format(prefix.rstrip(".")))
            # In version < 3, running_var are used without +eps.
            state_dict[prefix + "running_var"] -= self.eps

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.
        Args:
            module (torch.nn.Module):
        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.
        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res
