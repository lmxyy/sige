import torch
from torch import nn
from torch.nn import functional as F

from .sub_mobile_spade_generator import SubMobileSPADEResnetBlock
from ..normalization import FusedSubMobileSPADE


class FusedSubMobileSPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, ic, opt, config):
        super(FusedSubMobileSPADEResnetBlock, self).__init__()
        # Attributes
        self.fin = fin
        self.fout = fout
        self.ic = ic
        self.opt = opt
        self.learned_shortcut = fin != fout
        self.ic = ic
        self.config = config
        channel, hidden = config["channel"], config["hidden"]
        self.nhidden = hidden

        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(ic, channel, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        else:
            self.conv_1 = nn.Conv2d(channel, ic, kernel_size=3, padding=1)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(ic, channel, kernel_size=1, bias=False)

        # define normalization layers
        spade_config_str = opt.norm_G
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(opt.semantic_nc, hidden * (3 if self.learned_shortcut else 2), kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.norm_0 = FusedSubMobileSPADE(spade_config_str, fin, nhidden=hidden, oc=ic)
        self.norm_1 = FusedSubMobileSPADE(spade_config_str, fmiddle, nhidden=hidden, oc=channel)
        if self.learned_shortcut:
            self.norm_s = FusedSubMobileSPADE(spade_config_str, fin, nhidden=hidden, oc=ic)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        seg = F.interpolate(seg, size=x.size()[2:], mode="nearest")
        actvs = self.mlp_shared(seg)
        if self.learned_shortcut:
            actv_0, actv_1, actv_s = torch.split(actvs, self.nhidden, dim=1)
        else:
            actv_0, actv_1 = torch.split(actvs, self.nhidden, dim=1)
            actv_s = None

        x_s = self.shortcut(x, actv_s)

        dx = self.conv_0(self.actvn(self.norm_0(x, actv_0)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, actv_1)))

        out = x_s + dx

        return out

    def shortcut(self, x, actv):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, actv))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

    @classmethod
    def from_SubMobileSPADEResnetBlock(cls, original_block: SubMobileSPADEResnetBlock):
        fin = original_block.fin
        fout = original_block.fout
        ic = original_block.ic
        opt = original_block.opt
        config = original_block.config
        ret: FusedSubMobileSPADEResnetBlock = cls(fin, fout, ic, opt, config)

        ret.conv_0 = original_block.conv_0
        ret.norm_0.param_free_norm = original_block.norm_0.param_free_norm
        ret.norm_0.mlp_gamma = original_block.norm_0.mlp_gamma
        ret.norm_0.mlp_beta = original_block.norm_0.mlp_beta

        ret.conv_1 = original_block.conv_1
        ret.norm_1.param_free_norm = original_block.norm_1.param_free_norm
        ret.norm_1.mlp_gamma = original_block.norm_1.mlp_gamma
        ret.norm_1.mlp_beta = original_block.norm_1.mlp_beta

        if ret.learned_shortcut:
            ret.conv_s = original_block.conv_s
            ret.norm_s.param_free_norm = original_block.norm_s.param_free_norm
            ret.norm_s.mlp_gamma = original_block.norm_s.mlp_gamma
            ret.norm_s.mlp_beta = original_block.norm_s.mlp_beta
            ret.mlp_shared[0].weight.data = torch.cat(
                [
                    original_block.norm_0.mlp_shared[0].weight.data,
                    original_block.norm_1.mlp_shared[0].weight.data,
                    original_block.norm_s.mlp_shared[0].weight.data,
                ],
                dim=0,
            )
            ret.mlp_shared[0].bias.data = torch.cat(
                [
                    original_block.norm_0.mlp_shared[0].bias.data,
                    original_block.norm_1.mlp_shared[0].bias.data,
                    original_block.norm_s.mlp_shared[0].bias.data,
                ],
                dim=0,
            )
        else:
            ret.mlp_shared[0].weight.data = torch.cat(
                [
                    original_block.norm_0.mlp_shared[0].weight.data,
                    original_block.norm_1.mlp_shared[0].weight.data,
                ],
                dim=0,
            )
            ret.mlp_shared[0].bias.data = torch.cat(
                [
                    original_block.norm_0.mlp_shared[0].bias.data,
                    original_block.norm_1.mlp_shared[0].bias.data,
                ],
                dim=0,
            )
        return ret


class FusedSubMobileSPADEGenerator(nn.Module):
    def __init__(self, opt, config, **kwargs):
        super(FusedSubMobileSPADEGenerator, self).__init__()
        self.opt = opt
        self.config = config
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        # downsampled segmentation map instead of random z
        channel = config["channels"][0]
        self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * channel, 3, padding=1)

        ic = channel * 16
        channel = config["channels"][1]
        self.head_0 = FusedSubMobileSPADEResnetBlock(
            16 * nf, 16 * nf, ic, opt, {"channel": channel * 16, "hidden": channel * 2}
        )

        channel = config["channels"][2]
        self.G_middle_0 = FusedSubMobileSPADEResnetBlock(
            16 * nf, 16 * nf, ic, opt, {"channel": channel * 16, "hidden": channel * 2}
        )

        channel = config["channels"][3]
        self.G_middle_1 = FusedSubMobileSPADEResnetBlock(
            16 * nf, 16 * nf, ic, opt, {"channel": channel * 16, "hidden": channel * 2}
        )

        channel = config["channels"][4]
        self.up_0 = FusedSubMobileSPADEResnetBlock(
            16 * nf, 8 * nf, ic, opt, {"channel": channel * 8, "hidden": channel * 2}
        )

        ic = channel * 8
        channel = config["channels"][5]
        self.up_1 = FusedSubMobileSPADEResnetBlock(
            8 * nf, 4 * nf, ic, opt, {"channel": channel * 4, "hidden": channel * 2}
        )
        ic = channel * 4
        channel = config["channels"][6]
        self.up_2 = FusedSubMobileSPADEResnetBlock(
            4 * nf, 2 * nf, ic, opt, {"channel": channel * 2, "hidden": channel * 2}
        )
        ic = channel * 2
        channel = config["channels"][7]
        self.up_3 = FusedSubMobileSPADEResnetBlock(2 * nf, 1 * nf, ic, opt, {"channel": channel, "hidden": channel * 2})

        final_nc = channel

        if opt.num_upsampling_layers == "most":
            raise NotImplementedError
        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == "normal":
            num_up_layers = 5
        elif opt.num_upsampling_layers == "more":
            num_up_layers = 6
        elif opt.num_upsampling_layers == "most":
            num_up_layers = 7
        else:
            raise ValueError("opt.num_upsampling_layers [%s] not recognized" % opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input):
        opt = self.opt
        seg = input

        # we downsample segmap and run convolution
        x = F.interpolate(seg, size=(self.sh, self.sw))
        x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if opt.num_upsampling_layers in ("more", "most"):
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == "most":
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x
