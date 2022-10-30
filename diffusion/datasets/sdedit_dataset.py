import os.path

from PIL import Image
from torch.utils.data import Dataset


class SDEditDataset(Dataset):
    def __init__(self, args, config, transform):
        super(SDEditDataset, self).__init__()
        self.args = args
        self.config = config
        self.transform = transform
        self.paths = []
        root = config.data.data_root
        original_root = os.path.join(root, "original")
        edited_root = os.path.join(root, "edited")
        files = sorted(os.listdir(original_root))
        if args.image_metas is None:
            for file in files:
                if file.endswith(".png"):
                    original_path = os.path.join(original_root, file)
                    edited_path = os.path.join(edited_root, file)
                    assert os.path.exists(original_path) and os.path.exists(edited_path)
                    self.paths.append((original_path, edited_path))
        else:
            metas = args.image_metas
            assert len(metas) > 0
            for meta in metas:
                file = meta + ".png"
                original_path = os.path.join(original_root, file)
                edited_path = os.path.join(edited_root, file)
                assert os.path.exists(original_path) and os.path.exists(edited_path)
                self.paths.append((original_path, edited_path))

    def __getitem__(self, index):
        original_path, edited_path = self.paths[index]
        original_x = Image.open(original_path)
        edited_x = Image.open(edited_path)
        if self.transform is not None:
            original_x = self.transform(original_x)
            edited_x = self.transform(edited_x)
        filename = os.path.basename(original_path)
        name = os.path.splitext(filename)[0]
        return original_x, edited_x, index, name

    def __len__(self):
        return len(self.paths)
