import torch
import torch.nn as nn
from packaging import version

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"


class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        if "cond_view" in c:
            out = self.diffusion_model(
                x,
                timesteps=t,
                context=c.get("crossattn", None),
                y=c.get("vector", None),
                cond_view=c.get("cond_view", None),
                cond_motion=c.get("cond_motion", None),
                **kwargs,
            )
            if hasattr(self, "args") and self.args.mode == "profile_unet":
                from torchprofile import profile_macs
                import time
                from tqdm import trange

                if not hasattr(self.diffusion_model, "mode") or self.diffusion_model.mode == "sparse":
                    if hasattr(self.diffusion_model, "mode"):
                        self.diffusion_model.set_mode("profile")

                    macs = profile_macs(self.diffusion_model, (x, t, c.get("crossattn", None), c.get("vector", None), 
                                                               c.get("cond_view", None), c.get("cond_motion", None)))
                    print(f"MACs: {macs / 1e9:.3f}G")

                    if hasattr(self.diffusion_model, "mode"):
                        self.diffusion_model.set_mode("sparse")
                    for _ in trange(100):
                        self.diffusion_model(
                            x,
                            timesteps=t,
                            context=c.get("crossattn", None),
                            y=c.get("vector", None),
                            cond_view=c.get("cond_view", None),
                            cond_motion=c.get("cond_motion", None),
                            **kwargs,
                        )
                        torch.cuda.synchronize()
                    start = time.time()
                    for _ in trange(100):
                        self.diffusion_model(
                            x,
                            timesteps=t,
                            context=c.get("crossattn", None),
                            y=c.get("vector", None),
                            cond_view=c.get("cond_view", None),
                            cond_motion=c.get("cond_motion", None),
                            **kwargs,
                        )
                        torch.cuda.synchronize()
                    print(f"Time per forward pass: {(time.time() - start) * 10:.2f} ms")
                    exit(0)
        else:
            out = self.diffusion_model(
                x,
                timesteps=t,
                context=c.get("crossattn", None),
                y=c.get("vector", None),
                **kwargs,
            )
            if hasattr(self, "args") and self.args.mode == "profile_unet":
                from torchprofile import profile_macs
                import time
                from tqdm import trange

                if not hasattr(self.diffusion_model, "mode") or self.diffusion_model.mode == "sparse":
                    if hasattr(self.diffusion_model, "mode"):
                        self.diffusion_model.set_mode("profile")

                    macs = profile_macs(self.diffusion_model, (x, t, c.get("crossattn", None), c.get("vector", None)))
                    print(f"MACs: {macs / 1e9:.3f}G")

                    if hasattr(self.diffusion_model, "mode"):
                        self.diffusion_model.set_mode("sparse")
                    for _ in trange(100):
                        self.diffusion_model(
                            x,
                            timesteps=t,
                            context=c.get("crossattn", None),
                            y=c.get("vector", None),
                            **kwargs,
                        )
                        torch.cuda.synchronize()
                    start = time.time()
                    for _ in trange(100):
                        self.diffusion_model(
                            x,
                            timesteps=t,
                            context=c.get("crossattn", None),
                            y=c.get("vector", None),
                            **kwargs,
                        )
                        torch.cuda.synchronize()
                    print(f"Time per forward pass: {(time.time() - start) * 10:.2f} ms")
                    exit(0)
        return out
