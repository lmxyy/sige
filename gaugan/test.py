import argparse

from runner import Runner


def get_args():
    parser = argparse.ArgumentParser()

    # Model related
    parser.add_argument("--netG", type=str, default="spade")
    parser.add_argument("--restore_from", type=str, default=None)
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--input_nc", type=int, default=35)
    parser.add_argument("--output_nc", type=int, default=3)
    parser.add_argument(
        "--separable_conv_norm",
        type=str,
        default="instance",
        choices=("none", "instance", "batch"),
        help="whether to use instance norm for the separable convolutions",
    )
    parser.add_argument(
        "--norm_G", type=str, default="spadesyncbatch3x3", help="instance normalization or batch normalization"
    )
    parser.add_argument(
        "--num_upsampling_layers",
        choices=("normal", "more", "most"),
        default="more",
        help="If 'more', adds upsampling layer between the two middle resnet blocks. "
        "If 'most', also add one more upsampling + resnet layer at the end of the generator",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="instance",
        help="instance normalization or batch normalization [instance | batch | none]",
    )
    parser.add_argument(
        "--config_str", type=str, default=None, help="the configuration string for a specific subnet in the supernet"
    )

    # SIGE related
    parser.add_argument("--main_block_size", type=int, default=6)
    parser.add_argument("--shortcut_block_size", type=int, default=4)
    parser.add_argument("--num_sparse_layers", type=int, default=5)
    parser.add_argument("--mask_dilate_radius", type=int, default=1)
    parser.add_argument("--downsample_dilate_radius", type=int, default=2)

    # Data related
    parser.add_argument("--data_root", type=str, default="database/cityscapes-edit")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--aspect_ratio", type=int, default=2)
    parser.add_argument("--num_workers", default=8, type=int, help="# workers for loading data")
    parser.add_argument("--no_instance", action="store_true")
    parser.add_argument("--no_symmetric_editing", action="store_true")
    parser.add_argument("--image_ids", type=int, nargs="+", default=None, help="specify image ids for test")

    # Profile related
    parser.add_argument("--warmup_times", type=int, default=200)
    parser.add_argument("--test_times", type=int, default=200)

    # Other
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--mode", type=str, default="generate", choices=("generate", "profile"))
    parser.add_argument("--dont_save_label", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--download_tool", type=str, default="torch_hub", choices=("gdown", "torch_hub"))
    args = parser.parse_args()
    args.semantic_nc = args.input_nc + (0 if args.no_instance else 1)
    return args


def main():
    args = get_args()
    runner = Runner(args)
    runner.run()


if __name__ == "__main__":
    main()
