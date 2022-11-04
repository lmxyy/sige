import argparse
import os
import zipfile

from download_helper import download

NAME = "church_outdoor_sdedit.zip"
BASE_URL = "https://www.cs.cmu.edu/~sige/resources/datasets/"
GDOWN_URL = "https://drive.google.com/u/0/uc?id=1lV1lSo5qYDCT5AnDb_TSdYUiardy890U"
MD5 = "149d28096a94cfb48210e5950ffa613a"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="database", help="root directory to save the dataset")
    parser.add_argument("--tool", type=str, default="torch_hub", choices=("torch_hub", "gdown"), help="download tool")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    root = args.root
    os.makedirs(os.path.join(root, os.path.splitext(NAME)[0]), exist_ok=True)
    path = os.path.join(root, NAME)

    tool = args.tool
    if tool == "torch_hub":
        url = os.path.join(BASE_URL, NAME)
    elif tool == "gdown":
        url = GDOWN_URL
    else:
        raise NotImplementedError("Unknown tool [%s]!!!" % tool)
    download(NAME, url, path, MD5, tool=tool)
    print("Finished downloading. Extracting...")
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(os.path.join(root, os.path.splitext(NAME)[0]))
