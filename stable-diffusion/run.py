import argparse

from utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="inpainting")
    args, _ = parser.parse_known_args()

    if args.task == "inpainting":
        from runners.inpainting_runner import InpaintingRunner as Runner
    elif args.task == "sdedit":
        from runners.sdedit_runner import SDEditRunner as Runner
    else:
        raise NotImplementedError("Unknown task [%s]!!!" % args.task)

    parser = Runner.modify_commandline_options(parser)
    args = parser.parse_args()
    runner = Runner(args)
    set_seed(args.seed)

    runner.run()


if __name__ == "__main__":
    main()
