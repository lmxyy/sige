import argparse

from cleanfid import fid


def get_fid(parser: argparse.ArgumentParser):
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()
    score = fid.compute_fid(args.root, dataset_name="lsun_church", dataset_res=256, dataset_split="trainfull")
    return score
