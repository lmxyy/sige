import argparse
import math
import os
import threading

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import drn, get_trainIds
from download_helper import download


class Normalize(object):
    """
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

    def __call__(self, image, label):
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        return image, label


class ToTensor(object):
    """
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic, label):
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == "YCbCr":
            nchannel = 3
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float() / 255
        return img, torch.from_numpy(np.array(label, dtype=np.int))


class Compose(object):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class SegList(Dataset):
    def __init__(self, args):
        self.args = args
        image_root, data_root, mask_root = args.image_root, args.data_root, args.mask_root
        self.image_paths, self.label_paths = [], []
        self.mask_paths = None if mask_root is None else []
        self.transforms = Compose(
            [
                ToTensor(),
                Normalize(
                    mean=[0.29010095242892997, 0.32808144844279574, 0.28696394422942517],
                    std=[0.1829540508368939, 0.18656561047509476, 0.18447508988480435],
                ),
            ]
        )
        meta_path = os.path.join(data_root, "meta.csv")
        with open(meta_path, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                continue
            line = line.strip().split(",")
            gt_id, synthetic_id = line[0], line[1]
            self.image_paths.append(os.path.join(image_root, "%s_synthetic.png") % synthetic_id)
            self.label_paths.append(os.path.join(data_root, "synthetic_labels", "%s.npy") % synthetic_id)
            self.image_paths.append(os.path.join(image_root, "%s_gt.png") % synthetic_id)
            self.label_paths.append(os.path.join(data_root, "gt_labels", "%s.npy") % gt_id)
            if mask_root is not None:
                self.mask_paths.append(os.path.join(mask_root, synthetic_id + ".npy"))
                self.mask_paths.append(os.path.join(mask_root, synthetic_id + ".npy"))

    def __getitem__(self, index):
        name = "%04d" % index
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]
        label = np.load(label_path)
        new_label = np.empty(label.shape, dtype=label.dtype)
        for l in get_trainIds.labels:
            new_label[label == l.id] = l.trainId
        data = [Image.open(image_path), new_label]
        data = list(self.transforms(*data))
        if self.mask_paths is not None:
            mask = torch.from_numpy(np.load(self.mask_paths[index]))
            data.append(mask)
        return tuple(data)

    def __len__(self):
        return len(self.image_paths)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None, pretrained=True, use_torch_up=False):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(pretrained=pretrained, num_classes=1000)

        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, classes, kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(
                classes, classes, 16, stride=8, padding=4, output_padding=0, groups=classes, bias=False
            )
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        return self.softmax(y), x

    def optim_parameters(self, memo=None):
        raise NotImplementedError("This code is just for evaluation!!!")


def fast_hist(pred, label, n, mask=None):
    k = (label >= 0) & (label < n)
    if mask is not None:
        k &= mask
    return np.bincount(n * label[k].astype(int) + pred[k], minlength=n**2).reshape(n, n)


def resize_4d_tensor(tensor, width, height):
    """
    tensor: the semantic label tensor of shape [B, C, H, W]
    width: target width
    height: target height
    """
    tensor_cpu = tensor.cpu().numpy()
    if tensor.size(2) == height and tensor.size(3) == width:
        return tensor_cpu
    out_size = (tensor.size(0), tensor.size(1), height, width)
    out = np.empty(out_size, dtype=np.float32)

    def resize_channel(j):
        for i in range(tensor.size(0)):
            out[i, j] = np.array(Image.fromarray(tensor_cpu[i, j]).resize((width, height), Image.BILINEAR))

    workers = [threading.Thread(target=resize_channel, args=(j,)) for j in range(tensor.size(1))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    return out


def get_miou(parser: argparse.ArgumentParser):
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="database/cityscapes-edit")
    parser.add_argument("--mask_root", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="pretrained/drn-d-105_ms_cityscapes.pth")
    parser.add_argument("--device", type=str, default=None, choices=("cpu", "cuda"))
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)

    dataset = SegList(args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    model = DRNSeg("drn_d_105", 19, pretrained=False)
    model.load_state_dict(
        torch.load(
            download(
                name=os.path.basename(args.model_path),
                url="https://drive.google.com/u/0/uc?id=1O-BH64Rvc40cT_SQrvbnebTWRwC74TJp",
                path=args.model_path,
                md5="22f6aac9dc4dcf255bad997463fd40d0",
                tool="gdown",
            )
        )
    )
    model.to(device)
    model.eval()
    hist = np.zeros((19, 19))
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            if args.mask_root is None:
                image, label = data
                mask = None
            else:
                image, label, mask = data
                mask = mask.flatten().cpu().numpy()
            image = image.to(device)
            final = model(image)[0]
            final = final.cpu().numpy()
            pred = final.argmax(axis=1)
            label = label.numpy()
            hist += fast_hist(pred.flatten(), label.flatten(), 19, mask=mask)
    ious = per_class_iu(hist) * 100
    return np.nanmean(ious)
