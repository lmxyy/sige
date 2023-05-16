import argparse
import sys

import yaml
from easydict import EasyDict
from PyQt5 import QtGui, QtWidgets

from ui.main_window import MainWindow
from utils import get_device, override_config, set_seed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17, help="random seed")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--hparams", type=str, default="", help="override hyperparameters")
    parser.add_argument("--device", type=str, default=None, choices=("cpu", "cuda", "mps"))
    parser.add_argument("--download_tool", type=str, default="torch_hub", choices=("gdown", "torch_hub"))
    args = parser.parse_args()
    return args


def parse_args_and_config():
    args = get_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    device = get_device(args.device)
    config.device = device
    override_config(args.hparams, config)

    set_seed(args.seed)
    return args, config


def main():
    args, config = parse_args_and_config()
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon("ui/icons/piecasso.ico"))
    window = MainWindow(args, config)
    app.exec_()


if __name__ == "__main__":
    main()
