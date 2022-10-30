import hashlib
import os
from typing import Optional

import gdown
import wget
import shutil

MD5_MAP = {
    "spade.pth": "21b8b5b29295b7208ba5cb48d818a367",
    "fused_spade.pth": "216677f8fe9e0c6564e3f17cbe55c695",
    "sub_mobile_spade-32_32_32_48_32_24_24_32.pth": "2f8373ff0433badbea664da90709ca3c",
    "sub_mobile_spade-16_40_16_8_16_16_8_16.pth": "2f3dd95928a2902d70d641ae74a9c563",
    "fused_sub_mobile_spade-32_32_32_48_32_24_24_32.pth": "530ff557d614df1378d0613f7a0865dc",
}

BASE_URL = "https://www.cs.cmu.edu/~sige/resources/models/gaugan"


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def download(name: str, url: str, path: str, md5: Optional[str] = None, tool: str = "wget"):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    need_download = True
    if os.path.exists(path):
        if md5 is not None:
            if md5_hash(path) == md5:
                need_download = False
            else:
                print("MD5 hash mismatch for [%s]: [%s] v.s. [%s]!!!" % (name, md5_hash(path), md5))
                print("Removing [%s] and downloading again!!!" % path)
                shutil.rmtree(path)
    if need_download:
        if tool == "wget":
            print("Downloading [%s] to [%s]..." % (url, path))
            wget.download(url, path)
        elif tool == "gdown":
            gdown.download(url, path)
        else:
            raise NotImplementedError("Unknown download tool [%s]!!!" % tool)


def get_ckpt_path(opt, root="pretrained", check=True):
    network = opt.netG
    if network == "spade":
        name = "spade.pth"
    elif network in ("fused_spade", "sige_fused_spade"):
        name = "fused_spade.pth"
    elif network == "sub_mobile_spade":
        if opt.config_str == "32_32_32_48_32_24_24_32":
            name = "sub_mobile_spade-32_32_32_48_32_24_24_32.pth"
        elif opt.config_str == "16_40_16_8_16_16_8_16":
            name = "sub_mobile_spade-16_40_16_8_16_16_8_16.pth"
        else:
            raise NotImplementedError("Unknown supported network config [%s]!!!" % opt.config_str)
    elif network in ("fused_sub_mobile_spade", "sige_fused_sub_mobile_spade"):
        if opt.config_str == "32_32_32_48_32_24_24_32":
            name = "fused_sub_mobile_spade-32_32_32_48_32_24_24_32.pth"
        else:
            raise NotImplementedError("Unknown supported network config [%s]!!!" % opt.config_str)
    else:
        raise NotImplementedError("Unknown network [%s]!!!" % network)
    path = os.path.join(root, name)
    download(name, os.path.join(BASE_URL, name), path, MD5_MAP[name] if check else None)
    return path
