import argparse
import os

import lpips
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm


def get_lpips(parser: argparse.ArgumentParser):
    parser.add_argument("--ref_root", type=str, default="database/church_outdoor_sdedit/gt")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--mask_root", type=str, default=None)
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "original"])
    parser.add_argument("-v", "--version", type=str, default="0.1")
    parser.add_argument("--net", type=str, default="alex", choices=("alex", "vgg"))
    parser.add_argument("--device", type=str, default=None, choices=("cuda", "cpu"))
    args = parser.parse_args()

    if args.mode == "gt":
        files = ["%03d" % i for i in range(423)]
    else:
        files = ["%03d" % i for i in range(454)]

    loss_fn = lpips.LPIPS(net=args.net, version=args.version, spatial=args.mask_root is not None)

    if args.device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)
    loss_fn.to(device)

    scores = []
    for file in tqdm(files):
        path1 = os.path.join(args.ref_root, file + ".png")
        path2 = os.path.join(args.root, file + ".png")
        img1 = lpips.im2tensor(lpips.load_image(path1))  # RGB image from [-1,1]
        img2 = lpips.im2tensor(lpips.load_image(path2))
        _, _, h1, w1 = img1.shape
        _, _, h2, w2 = img2.shape
        h, w = min(h1, h2), min(w1, w2)
        img1 = F.interpolate(img1, (h, w))
        img2 = F.interpolate(img2, (h, w))
        img1 = img1.to(device)
        img2 = img2.to(device)
        if args.mask_root is not None:
            path = os.path.join(args.mask_root, file + ".npy")
            mask = torch.from_numpy(np.load(path)).to(device)[None, None]
            mask = F.interpolate(mask.float(), (h, w)) > 0.3
            mask = mask
        else:
            mask = None
        with torch.no_grad():
            d = loss_fn.forward(img1, img2)
            if mask is not None:
                d = (d * mask).sum()
                d = (d / mask.sum())
            scores.append(d.item())

    return sum(scores) / len(scores)
