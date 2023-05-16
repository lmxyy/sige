import argparse

import torch
from einops import repeat

from ldm.models.sige_autoencoder import SIGEAutoencoderKL
from sige.nn import SIGEModel
from sige.utils import compute_difference_mask, dilate_mask, downsample_mask
from utils import load_img
from .base_runner import BaseRunner


class SDEditRunner(BaseRunner):
    @staticmethod
    def modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = super(SDEditRunner, SDEditRunner).modify_commandline_options(parser)
        parser.add_argument("--strength", type=float, default=0.8)
        parser.add_argument("--edited_img", type=str, required=True)
        return parser

    def __init__(self, args):
        super().__init__(args)

    def generate(self):
        is_sige_model = (
            self.config.model.params.unet_config.target == "ldm.modules.diffusionmodules.sige_openaimodel.SIGEUNetModel"
        )

        args = self.args
        model = self.model
        sampler = self.sampler
        device = self.device

        prompts = [args.prompt]
        uc = None
        if args.scale != 1.0:
            uc = model.get_learned_conditioning([""])
        c = model.get_learned_conditioning(prompts)

        edited_img = load_img(args.edited_img).to(device)
        edited_img = repeat(edited_img, "1 ... -> b ...", b=1)
        if args.init_img is not None:
            init_img = load_img(args.init_img).to(device)
            init_img = repeat(init_img, "1 ... -> b ...", b=1)
            difference_mask = compute_difference_mask(init_img, edited_img)
            print("Edit Ratio: %.2f%%" % (difference_mask.sum() / difference_mask.numel() * 100))
            difference_mask = dilate_mask(difference_mask, 5)
            masks = downsample_mask(difference_mask, min_res=(4, 4), dilation=1)
        else:
            init_img = None
            masks = None

        if isinstance(model.first_stage_model, SIGEAutoencoderKL):
            assert isinstance(model.first_stage_model.encoder, SIGEModel)
            assert init_img is not None, "Must provide an initial image for SIGE model"
            init_img = load_img(args.init_img).to(device)
            init_img = repeat(init_img, "1 ... -> b ...", b=1)
            model.first_stage_model.encoder.set_mode("full")
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_img))
            model.first_stage_model.encoder.set_mode("sparse")
            model.first_stage_model.encoder.set_masks(masks)
            edited_latent = model.get_first_stage_encoding(model.encode_first_stage(edited_img))
        else:
            edited_img = load_img(args.edited_img).to(device)
            edited_img = repeat(edited_img, "1 ... -> b ...", b=1)
            init_latent = None
            edited_latent = model.get_first_stage_encoding(model.encode_first_stage(edited_img))

        del model.first_stage_model.encoder
        model.first_stage_model.encoder = None
        torch.cuda.empty_cache()

        assert 0.0 <= args.strength <= 1.0, "can only work with strength in [0.0, 1.0]"
        t_enc = int(args.strength * args.ddim_steps)
        print(f"target t_enc is {t_enc} steps")

        noise = torch.randn_like(edited_latent)

        z_enc_edited = sampler.stochastic_encode(edited_latent, torch.tensor([t_enc]).to(device), noise=noise)
        if is_sige_model:
            z_enc_init = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(device), noise=noise)
            samples_init, samples_edited = sampler.sige_img2img_decode(
                z_enc_init,
                z_enc_edited,
                c,
                t_enc,
                masks=masks,
                unconditional_guidance_scale=args.scale,
                unconditional_conditioning=uc,
            )
            samples = samples_edited
        else:
            samples_init = None
            samples = sampler.decode(
                z_enc_edited, c, t_enc, unconditional_guidance_scale=args.scale, unconditional_conditioning=uc
            )
        # samples = model.decode_first_stage(samples)
        if isinstance(model.first_stage_model, SIGEAutoencoderKL):
            difference_mask = dilate_mask(difference_mask, 40)
            masks = downsample_mask(difference_mask, min_res=(4, 4), dilation=0)
            assert isinstance(model.first_stage_model.decoder, SIGEModel)
            model.first_stage_model.decoder.set_mode("full")
            model.decode_first_stage(samples_init)
            model.first_stage_model.decoder.set_masks(masks)
            model.first_stage_model.decoder.set_mode("sparse")
        samples = model.decode_first_stage(samples)
        self.save_samples(samples)
