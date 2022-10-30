import argparse

from cleanfid import fid


def get_fid(parser: argparse.ArgumentParser):
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--gt_root", type=str, default="database/cityscapes-edit/images")
    args = parser.parse_args()
    score = fid.compute_fid(args.root, args.gt_root)
    return score
