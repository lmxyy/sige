import argparse

import numpy as np
import torch
from torchprofile import profile_macs

from sige.nn import Gather, Scatter, SIGEConv2d, SIGEModel, SIGEModule


class ExampleModule(SIGEModule):
    """
    A module consisting of a single `Gather`, 3x3 conv and `Scatter`.
    `SIGEModule` is a `nn.Module` wrapper that supports inference with three different modes:
    * `full`: The original inference. For the example above, the full mode will just perform the stardard $3 \times 3$ convolution.
    * `sparse`: The tiling-based sparse convolution.
    * `profile`: This mode is only used when profiling the MACs of the tiling-based convolution.
    It also supports setting the difference mask.

    `Gather`, `Scatter` and `SIGEConv2d` are also `SIGEModule`. Specifically,
    * `Gather` initialization requires the paired convolution and the sparse block size. During `full` inference, it will just record the input shape. During `sparse` inference, it will gather the active blocks according to the `active_indices` reduced from the difference mask. During `profile` inference, it will just create a dummy tensor to symbolicly track the computation graph for MACs profiling.
    * `Scatter` initialization requires the paired `Gather` module. During `full` inference, it will just cache the input tensor. During `sparse` inference, it will scatter the input blocks to the cached tensor according the `active_indices` in the paired `Gather`. During `profile` inference, it will just create a dummy tensor to symbolicly track the computation graph for MACs profiling.
    * `SIGEConv2d` is just a wrapper of `nn.Conv2d`. During `full` inference, it performs as the standard convolution. During `sparse` or `profile` inference, the `padding` will be 0 as the gathered blocks are already padded.
    """

    def __init__(self):
        super(ExampleModule, self).__init__()
        self.conv = SIGEConv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.gather = Gather(self.conv, block_size=6)
        self.scatter = Scatter(self.gather)

    def forward(self, x):
        x = self.gather(x)
        x = self.conv(x)
        x = self.scatter(x)
        return x


class ExampleModel(SIGEModel):
    # `SIGEModel` is a class to wrap the toppest `nn.Module`.
    # It supports setting difference masks and the inference mode to its children `SIGEModule`.
    def __init__(self):
        super(ExampleModel, self).__init__()
        self.example_module = ExampleModule()

    def forward(self, x: torch.Tensor):
        return self.example_module(x)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"], help="which device to use")
    return parser.parse_args()


def main():
    args = get_args()
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Get the inputs
    original_input = torch.randn((1, 16, 256, 256), device=device)
    mask = np.load("assets/mask.npy")
    mask = torch.from_numpy(mask).to(device)
    edited_input = original_input + torch.randn((1, 16, 256, 256), device=device) * mask[None, None]

    # Get the model
    model = ExampleModel().to(device)
    model.eval()

    with torch.no_grad():
        # Get the full model results
        model.set_mode("full")
        std_output = model(edited_input)
        full_macs = profile_macs(model, (edited_input,))

        # Cache the original input results
        model.set_mode("full")
        original_output = model(original_input)

        # SIGE Sparse Inference
        model.set_mode("sparse")
        # Set the differentiable mask
        model.set_masks({(256, 256): mask})  # The key is the resolution tuple and the value is the 2D mask tensor.
        sige_output = model(edited_input)
        model.set_mode("profile")
        sige_macs = profile_macs(model, (edited_input,))
        print("Max Error: %.6f" % abs(std_output - sige_output).max().item())
        assert torch.isclose(std_output, sige_output, atol=1e-4).all()
        print("Masked Region: %.2f%%" % (mask.sum() / mask.numel() * 100).item())
        print("Full MACs: %.2fM" % (full_macs / 1e6))
        print("SIGE MACs: %.2fM" % (sige_macs / 1e6))


if __name__ == "__main__":
    main()
