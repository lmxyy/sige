import argparse

import torch
import yaml
from easydict import EasyDict

from runner import Runner
from utils import override_config, set_seed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--hparams", type=str, default="", help="override hyperparameters")
    parser.add_argument("--restore_from", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=("cpu", "cuda"))
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--mode", type=str, default="generate", choices=("generate", "profile"))
    parser.add_argument("--image_metas", type=str, nargs="+", default=None, help="specify image names for test")

    # Profile related
    parser.add_argument("--warmup_times", type=int, default=200)
    parser.add_argument("--test_times", type=int, default=200)

    parser.add_argument("--download_tool", type=str, default="torch_hub", choices=("gdown", "torch_hub"))
    args = parser.parse_args()
    return args


def parse_args_and_config():
    args = get_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    if args.device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        assert args.device == "cuda"
        device = torch.device("cuda")
    config.device = device
    override_config(args.hparams, config)

    set_seed(args.seed)
    return args, config


def main():
    args, config = parse_args_and_config()
    runner = Runner(args, config)
    runner.run()


if __name__ == "__main__":
    main()
