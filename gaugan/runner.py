import time
import warnings
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchprofile import profile_macs

from cityscapes_dataset import CityscapesDataset
from download_helper import get_ckpt_path
from sige.nn import SIGEModel
from sige.utils import compute_difference_mask, dilate_mask, downsample_mask
from utils import decode_config, load_network, mytqdm, save_visuals


class Runner:
    def get_model(self) -> nn.Module:
        opt = self.opt
        netG = opt.netG

        config = None
        if netG == "spade":
            from gaugan.models.spade_generators.spade_generator import SPADEGenerator as Model
        elif netG == "fused_spade":
            from gaugan.models.spade_generators.fused_spade_generator import FusedSPADEGenerator as Model
        elif netG == "sige_fused_spade":
            from gaugan.models.spade_generators.sige_fused_spade_generator import SIGEFusedSPADEGenerator as Model
        elif netG == "sub_mobile_spade":
            from gaugan.models.sub_mobile_spade_generators.sub_mobile_spade_generator import (
                SubMobileSPADEGenerator as Model,
            )

            config = decode_config(opt.config_str)
        elif netG == "fused_sub_mobile_spade":
            from gaugan.models.sub_mobile_spade_generators.fused_sub_mobile_spade_generator import (
                FusedSubMobileSPADEGenerator as Model,
            )

            config = decode_config(opt.config_str)
        elif netG == "sige_fused_sub_mobile_spade":
            from gaugan.models.sub_mobile_spade_generators.sige_fused_sub_mobile_spade_generator import (
                SIGEFusedSubMobileSPADEGenerator as Model,
            )

            config = decode_config(opt.config_str)
        else:
            raise NotImplementedError("Unknown netG: [%s]!!!" % netG)

        model = Model(opt, config=config)

        # load network
        if opt.use_pretrained:
            pretrained_path = get_ckpt_path(opt, tool=opt.download_tool)
            if opt.restore_from is not None:
                warnings.warn("The model path will be overriden to [%s]!!!" % pretrained_path)
            opt.restore_from = pretrained_path
        if opt.restore_from is not None:
            model = load_network(model, opt.restore_from)
        model = model.to(opt.device)
        model.eval()

        return model

    def get_dataloader(self) -> DataLoader:
        opt = self.opt
        dataset = CityscapesDataset(opt)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)
        return dataloader

    def __init__(self, opt):
        self.opt = opt
        self.device = opt.device
        self.model = self.get_model()
        self.model.eval()
        self.dataloader = self.get_dataloader()

    def get_edges(self, t) -> torch.Tensor:
        edge = torch.zeros(t.size(), dtype=torch.uint8, device=self.device)
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | ((t[:, :, :, 1:] != t[:, :, :, :-1]).byte())
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | ((t[:, :, :, 1:] != t[:, :, :, :-1]).byte())
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | ((t[:, :, 1:, :] != t[:, :, :-1, :]).byte())
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | ((t[:, :, 1:, :] != t[:, :, :-1, :]).byte())
        return edge.float()

    def preprocess_input(self, data: Dict) -> torch.Tensor:
        opt = self.opt
        data["label"] = data["label"].long().to(opt.device)
        data["instance"] = data["instance"].to(opt.device)

        # create one-hot label map
        label_map = data["label"]
        b, c, h, w = label_map.shape
        assert c == 1
        c = opt.input_nc
        input_label = torch.zeros([b, c, h, w], device=opt.device)
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            instance_map = data["instance"]
            instance_edge_map = self.get_edges(instance_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        return input_semantics

    def generate(self):
        opt = self.opt
        if opt.save_dir is None:
            warnings.warn("The save_dir is not specified!!!")

        dataloader = self.dataloader
        model = self.model
        pbar_dataloader = mytqdm(dataloader, position=0, desc="Batch      ", leave=False)
        with torch.no_grad():
            for data in pbar_dataloader:
                original_label = data["original_label"]
                original_instance = data["original_instance"]
                edited_label = data["edited_label"]
                edited_instance = data["edited_instance"]
                data["label"] = torch.cat((original_label, edited_label), dim=0)
                data["instance"] = torch.cat((original_instance, edited_instance), dim=0)
                input_semantics = self.preprocess_input(data)
                if isinstance(model, SIGEModel):
                    difference_mask = compute_difference_mask(input_semantics[0], input_semantics[1], eps=1e-3)
                    difference_mask = dilate_mask(difference_mask, opt.mask_dilate_radius)
                    model.set_mode("full")
                    model(input_semantics[:1])
                    masks = downsample_mask(
                        difference_mask, (model.sh, model.sw), dilation=opt.downsample_dilate_radius
                    )
                    model.set_masks(masks)
                    if opt.verbose:
                        editing_ratio = float(difference_mask.sum() / difference_mask.numel())
                        message = "Image %s: Sparsity %.3f%%" % (data["name"][0], 100 * editing_ratio)
                        pbar_dataloader.write(message + "\n")
                    model.set_mode("sparse")
                    generated = model(input_semantics[1:])
                else:
                    generated = model(input_semantics[1:])
                name = data["name"][0]
                visuals = {"generated": generated[0]}
                if not opt.dont_save_label:
                    visuals["original_label"] = original_label[0]
                    visuals["edited_label"] = edited_label[0]
                if opt.save_dir is not None:
                    save_visuals(opt, visuals, name)

    def profile(self):
        opt = self.opt
        dataloader = self.dataloader
        model = self.model
        pbar_dataloader = mytqdm(dataloader, position=0, desc="Batch      ", leave=False)
        with torch.no_grad():
            for data in pbar_dataloader:
                original_label = data["original_label"]
                original_instance = data["original_instance"]
                edited_label = data["edited_label"]
                edited_instance = data["edited_instance"]
                data["label"] = torch.cat((original_label, edited_label), dim=0)
                data["instance"] = torch.cat((original_instance, edited_instance), dim=0)
                input_semantics = self.preprocess_input(data)
                if isinstance(model, SIGEModel):
                    difference_mask = compute_difference_mask(input_semantics[0], input_semantics[1], eps=1e-3)
                    difference_mask = dilate_mask(difference_mask, opt.mask_dilate_radius)
                    model.set_mode("full")
                    model(input_semantics[:1])
                    masks = downsample_mask(
                        difference_mask, (model.sh, model.sw), dilation=opt.downsample_dilate_radius
                    )
                    model.set_masks(masks)
                    editing_ratio = float(difference_mask.sum() / difference_mask.numel())

                    model.set_mode("profile")
                    macs = profile_macs(model, (input_semantics[:1],))
                    model.set_mode("sparse")
                    message = "Image %s: Sparsity %.3f%%    MACs %.3fG" % (
                        data["name"][0],
                        100 * editing_ratio,
                        macs / 1e9,
                    )
                else:
                    macs = profile_macs(model, (input_semantics[:1],))
                    message = "Image %s: MACs %.3fG" % (data["name"][0], macs / 1e9)
                for _ in mytqdm(range(opt.warmup_times), position=1, desc="Warmup     ", leave=False):
                    model(input_semantics[:1])
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                start_time = time.time()
                for _ in mytqdm(range(opt.test_times), position=1, desc="Measure    ", leave=False):
                    model(input_semantics[:1])
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                cost_time = time.time() - start_time
                message += "    Cost Time %.3fs    Avg Time %.3fms" % (cost_time, cost_time / opt.test_times * 1000)
                pbar_dataloader.write(message + "\n")

    def run(self):
        opt = self.opt
        if opt.mode == "generate":
            self.generate()
        elif opt.mode == "profile":
            self.profile()
        else:
            raise NotImplementedError("Unknown mode: [%s]!!!" % opt.mode)
