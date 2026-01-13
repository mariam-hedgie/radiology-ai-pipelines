import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import resize, InterpolationMode


class LungSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, image_size: int = 224):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_size = int(image_size)

        img_files = os.listdir(img_dir)
        mask_files = os.listdir(mask_dir)

        # Build map: key -> mask filename
        self.mask_map = {}
        for m in mask_files:
            stem = os.path.splitext(m)[0]
            stem = stem.replace("_mask", "")
            self.mask_map[stem] = m

        # Keep only images that have a matching mask
        self.images = []
        for img in img_files:
            stem = os.path.splitext(img)[0]
            if stem in self.mask_map:
                self.images.append(img)

        self.images = sorted(self.images)

        if len(self.images) == 0:
            raise RuntimeError("No matching image-mask pairs found!")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        stem = os.path.splitext(img_name)[0]
        mask_name = self.mask_map[stem]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # --- image ---
        img = Image.open(img_path).convert("RGB")
        img = resize(img, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        img = TF.to_tensor(img)

        # --- mask ---
        mask = Image.open(mask_path).convert("L")
        mask = resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()

        # binarize: background=0, lung=1
        mask = (mask > 0).long()

        return img, mask