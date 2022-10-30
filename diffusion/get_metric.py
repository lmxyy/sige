import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default="fid", choices=("fid", "psnr", "lpips"))
    args, _ = parser.parse_known_args()
    if args.metric == "fid":
        from metrics.fid_score import get_fid

        fid = get_fid(parser)
        print("FID: %.2f" % fid)
    elif args.metric == "psnr":
        from metrics.psnr_score import get_psnr

        psnr, std = get_psnr(parser)
        print("PSNR: %.2f +/- %.2f" % (psnr, std))
    elif args.metric == "lpips":
        from metrics.lpips_score import get_lpips

        lpips = get_lpips(parser)
        print("LPIPS: %.4f" % lpips)
    else:
        raise NotImplementedError("Unknown metric [%s]!!!" % args.metric)


if __name__ == "__main__":
    main()
