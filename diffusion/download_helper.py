import hashlib
import os
from typing import Optional

from torch.hub import download_url_to_file

MD5_MAP = {
    "church128-pd-unet.pth": "8a2690eb25d1e2b5367e9574d7b73822",
    "church256-ddim-fused_unet.pth": "fb3e26e5e6be812d6bed5c48e5265506",
    "church256-ddim-unet.pth": "8fbd43a27e98e64b56453faca9be619e",
}

BASE_URL = "https://www.cs.cmu.edu/~sige/resources/models/diffusion"


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def download(name: str, url: str, path: str, md5: Optional[str] = None, tool: str = "torch"):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    need_download = True
    if os.path.exists(path):
        if md5 is not None:
            if md5_hash(path) == md5:
                need_download = False
            else:
                print("MD5 hash mismatch for [%s]: [%s] v.s. [%s]!!!" % (name, md5_hash(path), md5))
                print("Removing [%s] and downloading again!!!" % path)
                os.remove(path)
    if need_download:
        if tool == "wget":
            print("Downloading [%s] to [%s]..." % (url, path))
            download_url_to_file(url, path)
        else:
            raise NotImplementedError("Unknown tool [%s]!!!" % tool)


def get_ckpt_path(config, root="pretrained", check=True):
    network = config.model.network
    if network == "ddim.unet":
        name = "church256-ddim-unet.pth"
    elif network in ["ddim.fused_unet", "ddim.sige_fused_unet"]:
        name = "church256-ddim-fused_unet.pth"
    elif network in ["pd.unet", "pd.sige_unet"]:
        name = "church128-pd-unet.pth"
    else:
        raise NotImplementedError("Unknown network [%s]!!!" % network)
    path = os.path.join(root, name)
    download(name, os.path.join(BASE_URL, name), path, MD5_MAP[name] if check else None)
    return path
