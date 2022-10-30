import os
import sys

from download_helper import download
import zipfile

NAME = "cityscapes-edit.zip"
BASE_URL = "https://www.cs.cmu.edu/~sige/resources/datasets"
MD5 = "f51ca362c4f7f0f733132ff9fe24613c"

if __name__ == "__main__":
    assert len(sys.argv) <= 2, "Usage: python download_dataset.py [root]"
    if len(sys.argv) == 2:
        root = sys.argv[1]
    else:
        root = "database"
    os.makedirs(os.path.join(root, os.path.splitext(NAME)[0]), exist_ok=True)
    path = os.path.join(root, NAME)
    download(NAME, os.path.join(BASE_URL, NAME), path, MD5)
    print('Finished downloading. Extracting...')
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(os.path.join(root, os.path.splitext(NAME)[0]))
