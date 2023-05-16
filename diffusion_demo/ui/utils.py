import numpy as np
import torch
from PyQt5 import QtGui


def pixmap2numpy(pixmap, inplace=False) -> np.ndarray:
    n_channels = 3
    image = pixmap.toImage()
    b = image.constBits()
    w, h = pixmap.width(), pixmap.height()
    b.setsize(h * w * 4)
    arr = np.frombuffer(b, np.uint8).reshape((h, w, 4))
    arr = arr[:, :, :n_channels]
    arr = arr[:, :, [2, 1, 0]]
    if not inplace:
        arr = arr.copy()

    return arr


def numpy2tensor(arr, inplace=False):
    arr = arr.astype(np.float32)
    if inplace:
        tensor = torch.from_numpy(arr).permute([2, 0, 1])  # [h, w, c]
    else:
        tensor = torch.tensor(arr).permute([2, 0, 1])  # [h, w, c]
    tensor = tensor.unsqueeze(0)  # [1, c, h, w]
    tensor = (tensor / 255 - 0.5) * 2
    return tensor


def pixmap2tensor(pixmap):
    arr = pixmap2numpy(pixmap)
    return numpy2tensor(arr)


def tensor2numpy(tensor):
    tensor = (tensor / 2 + 0.5) * 255
    tensor = tensor[0].permute([1, 2, 0])  # [h, w, c]
    arr = tensor.numpy()
    return arr


def numpy2pixmap(arr):
    arr = arr.astype(np.uint32)
    h, w, c = arr.shape
    b = (255 << 24 | arr[:, :, 0] << 16 | arr[:, :, 1] << 8 | arr[:, :, 2]).flatten()
    im = QtGui.QImage(b, w, h, QtGui.QImage.Format_RGB32)
    return QtGui.QPixmap.fromImage(im)


def tensor2pixmap(tensor):  # [1, c, h, w]
    arr = tensor2numpy(tensor)
    pixmap = numpy2pixmap(arr)
    return pixmap
