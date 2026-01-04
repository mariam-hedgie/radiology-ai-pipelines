import os
import shutil
import random
from pathlib import Path

# -------- CONFIG --------
DATA_ROOT = Path("data/lung_seg")
IMAGES_DIR = DATA_ROOT / "raw" / "images"
MASKS_DIR = DATA_ROOT / "raw" / "masks"

SPLITS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}

RANDOM_SEED = 42
# ------------------------

def get_id(fname):
    """
    Extract base ID used to match image and mask.
    Example:
      image: CHNCXR_0001_1.png
      mask:  CHNCXR_0001_1_mask.png
    """
    return fname.replace("_mask", "").split(".")[0]

def main():
    random.seed(RANDOM_SEED)

    images = sorted(os.listdir(IMAGES_DIR))
    masks = sorted(os.listdir(MASKS_DIR))

    # Build ID → file maps
    image_map = {get_id(f): f for f in images}
    mask_map = {get_id(f): f for f in masks}

    common_ids = sorted(set(image_map) & set(mask_map))
    print(f"Found {len(common_ids)} paired image-mask samples")

    random.shuffle(common_ids)

    n_total = len(common_ids)
    n_train = int(SPLITS["train"] * n_total)
    n_val = int(SPLITS["val"] * n_total)

    split_ids = {
        "train": common_ids[:n_train],
        "val": common_ids[n_train:n_train + n_val],
        "test": common_ids[n_train + n_val:],
    }

    for split, ids in split_ids.items():
        img_out = DATA_ROOT / split / "images"
        mask_out = DATA_ROOT / split / "masks"
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        for sid in ids:
            shutil.copy(IMAGES_DIR / image_map[sid], img_out / image_map[sid])
            shutil.copy(MASKS_DIR / mask_map[sid], mask_out / mask_map[sid])

        print(f"{split}: {len(ids)} samples")

if __name__ == "__main__":
    main()