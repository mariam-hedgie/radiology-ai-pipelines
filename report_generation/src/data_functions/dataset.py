# src/data_functions/dataset.py
import json
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image


class JsonlImageTextDataset(Dataset):
    """
    Expects JSONL with fields:
      - image: relative path from data_root
      - text: report text
      - uid: optional
    """

    def __init__(self, data_root: str, jsonl_name: str):
        self.data_root = Path(data_root)
        self.jsonl_path = self.data_root / jsonl_name

        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Missing JSONL: {self.jsonl_path}")

        self.records = []
        with open(self.jsonl_path) as f:
            for line in f:
                self.records.append(json.loads(line))

        if len(self.records) == 0:
            raise RuntimeError(f"No samples loaded from {self.jsonl_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]

        img_path = self.data_root / r["image"]
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")

        image = Image.open(img_path).convert("RGB")

        return {
            "image": image,
            "text": r["text"],
            "path": r.get("image", ""),
            "uid": r.get("uid", ""),
        }