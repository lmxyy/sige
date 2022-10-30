import argparse
import os

import numpy as np
import skimage
import torch
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm


def get_psnr(parser: argparse.ArgumentParser):
    parser.add_argument("--ref_root", type=str, default="database/church_outdoor_sdedit/gt")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--mask_root", type=str, default=None)
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "original"])
    args = parser.parse_args()

    if args.mode == "gt":
        files = ["%03d" % i for i in range(423)]
    else:
        files = ["%03d" % i for i in range(454)]

    psnrs = []
    for file in tqdm(files):
        path1 = os.path.join(args.ref_root, file + ".png")
        path2 = os.path.join(args.root, file + ".png")
        image1 = Image.open(path1)
        h1, w1 = image1.size
        image2 = Image.open(path2)

        h2, w2 = image2.size
        h = min(h1, h2)
        w = min(w1, w2)

        if args.mask_root is not None:
            path = os.path.join(args.mask_root, file + ".npy")
            mask = np.load(path)
            mask = torch.from_numpy(mask)[None, None]
            mask = F.interpolate(mask.float(), (h, w)) > 0.3
            mask = mask.numpy()
        else:
            mask = None

        image1 = image1.resize((h, w), resample=Image.NEAREST)
        image2 = image2.resize((h, w), resample=Image.NEAREST)
        image1 = np.array(image1)
        image2 = np.array(image2)

        if mask is not None:
            mask = mask.reshape(-1)
            image1 = image1.reshape(-1, 3)
            image2 = image2.reshape(-1, 3)
            image1 = image1[mask][None]
            image2 = image2[mask][None]
        psnr = skimage.metrics.peak_signal_noise_ratio(image1, image2)
        psnrs.append(psnr)
    psnrs = np.array(psnrs)
    return psnrs.mean(), psnrs.std()
