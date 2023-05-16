import time
from typing import Optional

import torch
from PyQt5 import QtCore, QtGui, QtWidgets

from runner import Runner
from ui.hparams import CANVAS_DIMENSIONS
from ui.utils import numpy2pixmap, pixmap2numpy


class InferenceThread(QtCore.QThread):
    change_pixmap_signal = QtCore.pyqtSignal(QtGui.QPixmap)

    def __init__(self, args, config, display_pad, main_window):
        super(InferenceThread, self).__init__()
        self.args = args
        self.config = config
        self.display_pad = display_pad
        self.main_window = main_window

        self.runner = Runner(args, config)
        self.run_flag = False
        self.original_img: Optional[torch.Tensor] = None
        self.edited_img: Optional[torch.Tensor] = None

    def run(self) -> None:
        start_time = time.time()
        self.main_window.statusBar.showMessage("Converting...")
        output_tensor = self.runner.generate(
            original_img=self.original_img, edited_img=self.edited_img, mode="sparse", sparse_update=self.sparse_update
        )
        output_numpy = output_tensor[0].permute([1, 2, 0]).numpy() * 255
        output_map = numpy2pixmap(output_numpy)
        elapsed_time = time.time() - start_time
        self.main_window.statusBar.showMessage("Conversion finished. Elapsed time: %.2fs." % elapsed_time)
        self.display_pad.output_image_tensor = output_tensor
        self.change_pixmap_signal.emit(output_map)

    def start_inference(self, original_img: torch.Tensor, edited_img: torch.Tensor, sparse_update=False):
        if sparse_update and not self.runner.is_sige_model():
            return
        self.original_img = original_img
        self.edited_img = edited_img
        self.sparse_update = sparse_update
        self.start()

    def reset_base_image(self, original_img: torch.Tensor):
        if not self.runner.is_sige_model():
            return
        start_time = time.time()
        self.main_window.statusBar.showMessage("Pre-computing...")
        self.runner.generate(original_img, original_img, mode="full", sparse_update=False)
        elapsed_time = time.time() - start_time
        self.main_window.statusBar.showMessage("Pre-computation finished. Elapsed time: %.2fs." % elapsed_time)


class DisplayPad(QtWidgets.QLabel):
    def __init__(self, main_window, canvas, args, config):
        super(DisplayPad, self).__init__()
        self.mode = "rectangle"
        self.canvas = canvas
        self.background_color = QtGui.QColor(QtCore.Qt.white)
        self.setPixmap(QtGui.QPixmap(*CANVAS_DIMENSIONS))
        self.pixmap().fill(self.background_color)
        self.args = args
        self.config = config
        self.inference_thread = InferenceThread(args, config, self, main_window)
        self.inference_thread.change_pixmap_signal.connect(self.display_image)
        self.base_image_tensor: Optional[torch.Tensor] = None
        self.edited_img_numpy = None
        self.output_image_tensor: Optional[torch.Tensor] = None

    def update(self, sparse_update=False):
        if sparse_update:
            edited_img_numpy = self.edited_img_numpy
            self.edited_img_numpy = None
        else:
            pixmap = self.canvas.pixmap()
            edited_img_numpy = pixmap2numpy(pixmap)
            self.edited_img_numpy = edited_img_numpy
        edited_img_tensor = torch.from_numpy(edited_img_numpy).permute(2, 0, 1).float() / 255.0
        edited_img_tensor = edited_img_tensor.unsqueeze(0).to(self.config.device)
        self.inference_thread.start_inference(self.base_image_tensor, edited_img_tensor, sparse_update=sparse_update)

    def set_base_image(self, tensor: torch.Tensor):
        tensor = tensor.to(self.config.device)
        self.base_image_tensor = tensor
        self.inference_thread.reset_base_image(tensor)

    @QtCore.pyqtSlot(QtGui.QPixmap)
    def display_image(self, pixmap):
        self.setPixmap(pixmap)
        if self.edited_img_numpy is not None:
            self.canvas.setPixmap(numpy2pixmap(self.edited_img_numpy))

    def apply(self):
        print("Apply changes")
        output_numpy = self.output_image_tensor[0].permute([1, 2, 0]).numpy() * 255
        self.update(sparse_update=True)
        output_map = numpy2pixmap(output_numpy)
        self.canvas.setPixmap(output_map)
        self.base_image_tensor = self.output_image_tensor.to(self.config.device)
        self.canvas.base_image_numpy = output_numpy
