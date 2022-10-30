import os

import numpy as np
import torch
from torch.utils.data import Dataset


class CityscapesDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        root = opt.data_root
        with open(os.path.join(root, "meta.csv"), "r") as f:
            lines = f.readlines()
        self.gt_ids, self.synthetic_ids = [], []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            id = i - 1
            if opt.image_ids is not None and id not in opt.image_ids:
                continue
            line = line.strip().split(",")
            self.gt_ids.append(line[0])
            self.synthetic_ids.append(line[1])

    def __len__(self):
        if self.opt.no_symmetric_editing:
            return len(self.gt_ids)
        else:
            return 2 * len(self.gt_ids)

    def __getitem__(self, idx):
        opt = self.opt
        total = len(self.gt_ids)
        if idx >= total:
            idx = idx - total
            original_type = "synthetic"
            edited_type = "gt"
        else:
            original_type = "gt"
            edited_type = "synthetic"
        original_id = getattr(self, original_type + "_ids")[idx]
        edited_id = getattr(self, edited_type + "_ids")[idx]

        original_label = np.load(os.path.join(opt.data_root, "%s_labels" % original_type, "%s.npy" % original_id))
        original_instance = np.load(os.path.join(opt.data_root, "%s_instances" % original_type, "%s.npy" % original_id))
        edited_label = np.load(os.path.join(opt.data_root, "%s_labels" % edited_type, "%s.npy" % edited_id))
        edited_instance = np.load(os.path.join(opt.data_root, "%s_instances" % edited_type, "%s.npy" % edited_id))
        image_path = os.path.join(opt.data_root, "images", "%s.png" % self.gt_ids[idx])

        # [h, w] -> [c, h, w]
        original_label = torch.from_numpy(original_label)[None]
        original_instance = torch.from_numpy(original_instance)[None]
        edited_label = torch.from_numpy(edited_label)[None]
        edited_instance = torch.from_numpy(edited_instance)[None]
        ret = {
            "original_label": original_label,
            "original_instance": original_instance,
            "edited_label": edited_label,
            "edited_instance": edited_instance,
            "image_path": image_path,
            "name": "%s_%s" % (self.synthetic_ids[idx], edited_type),
        }
        return ret
