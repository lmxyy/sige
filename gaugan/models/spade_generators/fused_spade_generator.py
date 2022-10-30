import torch
from torch import nn
from torch.nn import functional as F

from .spade_generator import SPADEResnetBlock
from ..normalization import FusedSPADE


class FusedSPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super(FusedSPADEResnetBlock, self).__init__()
        self.fin = fin
        self.fout = fout
        self.opt = opt
        # Attributes
        self.nhidden = opt.ngf * 2
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        # create conv layers
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(opt.semantic_nc, self.nhidden * (3 if self.learned_shortcut else 2), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # define normalization layers
        if "spectral" in opt.norm_G:
            raise NotImplementedError
        spade_config_str = opt.norm_G.replace("spectral", "")

        self.norm_0 = FusedSPADE(spade_config_str, fin, nhidden=opt.ngf * 2)
        self.norm_1 = FusedSPADE(spade_config_str, fmiddle, nhidden=opt.ngf * 2)
        if self.learned_shortcut:
            self.norm_s = FusedSPADE(spade_config_str, fin, nhidden=opt.ngf * 2)

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
    def from_SPADEResnetBlock(cls, spade_resnet_block: SPADEResnetBlock):
        fin = spade_resnet_block.fin
        fout = spade_resnet_block.fout
        opt = spade_resnet_block.opt
        ret: FusedSPADEResnetBlock = cls(fin, fout, opt)

        ret.conv_0 = spade_resnet_block.conv_0
        ret.norm_0.param_free_norm = spade_resnet_block.norm_0.param_free_norm
        ret.norm_0.mlp_gamma_beta.weight.data = torch.cat(
            [
                spade_resnet_block.norm_0.mlp_gamma.weight.data,
                spade_resnet_block.norm_0.mlp_beta.weight.data,
            ],
            dim=0,
        )
        ret.norm_0.mlp_gamma_beta.bias.data = torch.cat(
            [
                spade_resnet_block.norm_0.mlp_gamma.bias.data,
                spade_resnet_block.norm_0.mlp_beta.bias.data,
            ],
            dim=0,
        )

        ret.conv_1 = spade_resnet_block.conv_1
        ret.norm_1.param_free_norm = spade_resnet_block.norm_1.param_free_norm
        ret.norm_1.mlp_gamma_beta.weight.data = torch.cat(
            [
                spade_resnet_block.norm_1.mlp_gamma.weight.data,
                spade_resnet_block.norm_1.mlp_beta.weight.data,
            ],
            dim=0,
        )
        ret.norm_1.mlp_gamma_beta.bias.data = torch.cat(
            [
                spade_resnet_block.norm_1.mlp_gamma.bias.data,
                spade_resnet_block.norm_1.mlp_beta.bias.data,
            ],
            dim=0,
        )

        if ret.learned_shortcut:
            ret.conv_s = spade_resnet_block.conv_s
            ret.norm_s.param_free_norm = spade_resnet_block.norm_s.param_free_norm
            ret.norm_s.mlp_gamma_beta.weight.data = torch.cat(
                [
                    spade_resnet_block.norm_s.mlp_gamma.weight.data,
                    spade_resnet_block.norm_s.mlp_beta.weight.data,
                ],
                dim=0,
            )
            ret.norm_s.mlp_gamma_beta.bias.data = torch.cat(
                [
                    spade_resnet_block.norm_s.mlp_gamma.bias.data,
                    spade_resnet_block.norm_s.mlp_beta.bias.data,
                ],
                dim=0,
            )

            ret.mlp_shared[0].weight.data = torch.cat(
                [
                    spade_resnet_block.norm_0.mlp_shared[0].weight.data,
                    spade_resnet_block.norm_1.mlp_shared[0].weight.data,
                    spade_resnet_block.norm_s.mlp_shared[0].weight.data,
                ],
                dim=0,
            )
            ret.mlp_shared[0].bias.data = torch.cat(
                [
                    spade_resnet_block.norm_0.mlp_shared[0].bias.data,
                    spade_resnet_block.norm_1.mlp_shared[0].bias.data,
                    spade_resnet_block.norm_s.mlp_shared[0].bias.data,
                ],
                dim=0,
            )
        else:
            ret.mlp_shared[0].weight.data = torch.cat(
                [
                    spade_resnet_block.norm_0.mlp_shared[0].weight.data,
                    spade_resnet_block.norm_1.mlp_shared[0].weight.data,
                ],
                dim=0,
            )
            ret.mlp_shared[0].bias.data = torch.cat(
                [
                    spade_resnet_block.norm_0.mlp_shared[0].bias.data,
                    spade_resnet_block.norm_1.mlp_shared[0].bias.data,
                ],
                dim=0,
            )

        return ret


class FusedSPADEGenerator(nn.Module):
    def __init__(self, opt, **kwargs):
        super(FusedSPADEGenerator, self).__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = FusedSPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = FusedSPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = FusedSPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = FusedSPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = FusedSPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = FusedSPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = FusedSPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == "most":
            self.up_4 = FusedSPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

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

    def remove_spectral_norm(self):
        self.head_0.remove_spectral_norm()
        self.G_middle_0.remove_spectral_norm()
        self.G_middle_1.remove_spectral_norm()

        self.up_0.remove_spectral_norm()
        self.up_1.remove_spectral_norm()
        self.up_2.remove_spectral_norm()
        self.up_3.remove_spectral_norm()

        if self.opt.num_upsampling_layers == "most":
            self.up_4.remove_spectral_norm()
