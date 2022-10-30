import torch
from torch import nn
from torch.nn import functional as F

from sige.nn import Gather, SIGEConv2d, SIGEModel, SIGEModule, Scatter, ScatterGather, ScatterWithBlockResidual
from ..sige_normalization import SIGEFusedSPADE


class SIGEFusedSPADEResnetBlock(SIGEModule):
    def __init__(self, fin, fout, opt, support_sparse: bool = False):
        super(SIGEFusedSPADEResnetBlock, self).__init__()
        self.fin = fin
        self.fout = fout
        self.opt = opt
        # Attributes
        self.nhidden = opt.ngf * 2
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        main_block_size = opt.main_block_size
        main_support_sparse = support_sparse and main_block_size is not None

        MainConv2d = SIGEConv2d if main_support_sparse else nn.Conv2d
        # create conv layers
        self.mlp_shared = nn.Sequential(
            MainConv2d(opt.semantic_nc, self.nhidden * (3 if self.learned_shortcut else 2), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_0 = MainConv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = MainConv2d(fmiddle, fout, kernel_size=3, padding=1)

        if main_support_sparse:
            self.seg_gather = Gather(self.mlp_shared[0], main_block_size)
            self.seg_scatter_gather = ScatterGather(self.seg_gather)
            self.main_gather = Gather(self.conv_0, main_block_size)
            self.main_scatter_gather = ScatterGather(self.main_gather)

        if self.learned_shortcut:
            shortcut_block_size = opt.shortcut_block_size
            shortcut_support_sparse = main_support_sparse and shortcut_block_size is not None
            ShortcutConv2d = SIGEConv2d if shortcut_support_sparse else nn.Conv2d
            self.conv_s = ShortcutConv2d(fin, fout, kernel_size=1, bias=False)
            if shortcut_support_sparse:
                self.shortcut_gather = Gather(self.conv_s, shortcut_block_size)
                self.scatter = ScatterWithBlockResidual(self.main_gather, self.shortcut_gather)
            elif main_support_sparse:
                self.scatter = Scatter(self.main_gather)
        else:
            shortcut_block_size = None
            shortcut_support_sparse = False
            if main_support_sparse:
                self.scatter = Scatter(self.main_gather)

        self.main_support_sparse = main_support_sparse
        self.shortcut_support_sparse = shortcut_support_sparse

        # define normalization layers
        if "spectral" in opt.norm_G:
            raise NotImplementedError("Spectral norm is not implemented for SIGEModel!!!")
        spade_config_str = opt.norm_G.replace("spectral", "")

        self.norm_0 = SIGEFusedSPADE(
            spade_config_str,
            fin,
            nhidden=opt.ngf * 2,
            seg_gather=self.seg_gather if main_support_sparse else None,
            main_block_size=main_block_size,
            shortcut_block_size=shortcut_block_size,
        )
        self.norm_1 = SIGEFusedSPADE(
            spade_config_str,
            fmiddle,
            nhidden=opt.ngf * 2,
            seg_gather=self.seg_gather if main_support_sparse else None,
            main_block_size=main_block_size,
            shortcut_block_size=shortcut_block_size,
        )
        if self.learned_shortcut:
            self.norm_s = SIGEFusedSPADE(
                spade_config_str,
                fin,
                nhidden=opt.ngf * 2,
                seg_gather=self.seg_gather if main_support_sparse else None,
                shortcut_conv=self.conv_s,
                main_block_size=main_block_size,
                shortcut_block_size=shortcut_block_size,
            )

    def forward(self, x, seg):
        if self.mode == "full":
            return self.full_forward(x, seg)
        elif self.mode in ("sparse", "profile"):
            return self.sparse_forward(x, seg)
        else:
            raise NotImplementedError("Unknown mode [%s]!!!" % self.mode)

    def full_forward(self, x, seg):
        main_support_sparse = self.main_support_sparse
        shortcut_support_sparse = self.shortcut_support_sparse

        seg = F.interpolate(seg, size=x.size()[2:], mode="nearest")

        if main_support_sparse:
            seg = self.seg_gather(seg)  # Just for record the seg shape
        actvs = self.mlp_shared(seg)
        if main_support_sparse:
            actvs = self.seg_scatter_gather(actvs)

        if self.learned_shortcut:
            actv_0, actv_1, actv_s = torch.split(actvs, self.nhidden, dim=1)
            if shortcut_support_sparse:
                x_s = self.shortcut_gather(x)  # Just for record the x_s shape
            x_s = self.conv_s(self.norm_s(x_s, actv_s))
        else:
            actv_0, actv_1 = torch.split(actvs, self.nhidden, dim=1)
            x_s = x

        dx = x
        if main_support_sparse:
            dx = self.main_gather(dx)
        dx = self.conv_0(self.actvn(self.norm_0(dx, actv_0)))
        if main_support_sparse:
            dx = self.main_scatter_gather(dx)
        dx = self.conv_1(self.actvn(self.norm_1(dx, actv_1)))

        if main_support_sparse:
            out = self.scatter(dx, x_s)
        else:
            out = x_s + dx

        return out

    def sparse_forward(self, x, seg):
        main_support_sparse = self.main_support_sparse
        shortcut_support_sparse = self.shortcut_support_sparse

        seg = F.interpolate(seg, size=x.size()[2:], mode="nearest")

        if main_support_sparse:
            seg = self.seg_gather(seg)  # Just for record the seg shape
        actvs = self.mlp_shared(seg)
        if main_support_sparse:
            actvs = self.seg_scatter_gather(actvs)

        if self.learned_shortcut:
            actv_0, actv_1, actv_s = torch.split(actvs, self.nhidden, dim=1)
            if shortcut_support_sparse:
                x_s = self.shortcut_gather(
                    x, self.norm_s.scale.view(1, -1, 1, 1), self.norm_s.shift.view(1, -1, 1, 1)
                )  # Just for record the x_s shape
            else:
                x_s = self.norm_s.param_free_norm(x)
            x_s = self.conv_s(self.norm_s(x_s, actv_s))
        else:
            actv_0, actv_1 = torch.split(actvs, self.nhidden, dim=1)
            x_s = x

        if main_support_sparse:
            dx = self.main_gather(x, self.norm_0.scale.view(1, -1, 1, 1), self.norm_0.shift.view(1, -1, 1, 1))
        else:
            dx = self.norm_0.param_free_norm(x)
        dx = self.conv_0(self.actvn(self.norm_0(dx, actv_0)))
        if main_support_sparse:
            dx = self.main_scatter_gather(dx, self.norm_1.scale.view(1, -1, 1, 1), self.norm_1.shift.view(1, -1, 1, 1))
        else:
            dx = self.norm_1.param_free_norm(dx)
        dx = self.conv_1(self.actvn(self.norm_1(dx, actv_1)))

        if main_support_sparse:
            out = self.scatter(dx, x_s)
        else:
            out = x_s + dx
        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SIGEFusedSPADEGenerator(SIGEModel):
    def __init__(self, opt, **kwargs):
        super(SIGEFusedSPADEGenerator, self).__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        is_most = opt.num_upsampling_layers == "most"
        num_sparse_layers = opt.num_sparse_layers

        self.head_0 = SIGEFusedSPADEResnetBlock(16 * nf, 16 * nf, opt, support_sparse=num_sparse_layers >= 7 + is_most)

        self.G_middle_0 = SIGEFusedSPADEResnetBlock(
            16 * nf, 16 * nf, opt, support_sparse=num_sparse_layers >= 6 + is_most
        )
        self.G_middle_1 = SIGEFusedSPADEResnetBlock(
            16 * nf, 16 * nf, opt, support_sparse=num_sparse_layers >= 5 + is_most
        )

        self.up_0 = SIGEFusedSPADEResnetBlock(16 * nf, 8 * nf, opt, support_sparse=num_sparse_layers >= 4 + is_most)
        self.up_1 = SIGEFusedSPADEResnetBlock(8 * nf, 4 * nf, opt, support_sparse=num_sparse_layers >= 3 + is_most)
        self.up_2 = SIGEFusedSPADEResnetBlock(4 * nf, 2 * nf, opt, support_sparse=num_sparse_layers >= 2 + is_most)
        self.up_3 = SIGEFusedSPADEResnetBlock(2 * nf, 1 * nf, opt, support_sparse=num_sparse_layers >= 1 + is_most)

        final_nc = nf

        if opt.num_upsampling_layers == "most":
            self.up_4 = SIGEFusedSPADEResnetBlock(1 * nf, nf // 2, opt, support_sparse=num_sparse_layers >= is_most)
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
