import torch
from torch import nn

from sige.nn import SIGEConv2d, SIGEModule


def my_instance_norm(x: torch.Tensor, norm: nn.InstanceNorm2d, no_norm: bool = False):
    assert norm.track_running_stats == False
    n, c, h, w = x.shape
    assert n == 1
    var, mean = torch.var_mean(x, unbiased=False, dim=[2, 3], keepdim=True)
    var = torch.sqrt(var + norm.eps)
    if not no_norm:
        x = (x - mean) / var
    scale = 1 / var[0, :, 0, 0]
    shift = -(mean / var)[0, :, 0, 0]
    if norm.affine:
        if not no_norm:
            x = x.view(n, c, h, w)
            x = x * norm.weight.view(1, -1, 1, 1)
            x = x + norm.bias.view(1, -1, 1, 1)
        scale = scale * norm.weight
        shift = shift * norm.weight
        shift = shift + norm.bias
    return x, scale, shift


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        norm_layer=nn.InstanceNorm2d,
        use_bias=True,
        scale_factor=1,
    ):
        super(SeparableConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels * scale_factor,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=use_bias,
            ),
            norm_layer(in_channels * scale_factor),
            nn.Conv2d(
                in_channels=in_channels * scale_factor,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=use_bias,
            ),
        )

    def forward(self, x):
        return self.conv(x)


class SIGESeparableConv2d(SIGEModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        norm_layer=nn.InstanceNorm2d,
        use_bias=True,
        scale_factor=1,
        support_sparse=False,
    ):
        super(SIGESeparableConv2d, self).__init__()
        self.support_sparse = support_sparse

        Conv2d = SIGEConv2d if support_sparse else nn.Conv2d
        self.conv = nn.Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=in_channels * scale_factor,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=use_bias,
            ),
            norm_layer(in_channels * scale_factor),
            Conv2d(
                in_channels=in_channels * scale_factor,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=use_bias,
            ),
        )

        self.scale, self.shift = None, None

    def forward(self, x):
        if self.mode == "full":
            assert x.shape[0] == 1
            x = self.conv[0](x)
            x, self.scale, self.shift = my_instance_norm(x, self.conv[1])
            x = self.conv[2](x)
        elif self.mode in ("sparse", "profile"):
            if self.support_sparse:
                x = self.conv[0](x)
                x = x * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)
                x = self.conv[2](x)
            else:
                x = self.conv(x)
        else:
            raise NotImplementedError("Unknown mode [%s]!!!" % self.mode)
        return x
