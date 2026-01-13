# src/datasets/vindr_rib_dataset.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from torchvision import transforms


Point = Tuple[float, float]
Polygon = List[Point]


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _parse_polygon(obj: Any) -> List[Polygon]:
    """
    Normalizes various JSON polygon formats into a list of polygons,
    where each polygon is a list of (x,y) points.

    Supported:
      1) [{"x":..., "y":...}, ...]                        -> one polygon
      2) [[{"x":..., "y":...}, ...], [...]]               -> many polygons
      3) [[x,y], [x,y], ...]                              -> one polygon
      4) [[[x,y], ...], [[x,y], ...]]                     -> many polygons
    """
    if obj is None:
        return []

    # Case 1: list of dict points => one polygon
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict) and "x" in obj[0] and "y" in obj[0]:
        poly = [(float(p["x"]), float(p["y"])) for p in obj]
        return [poly] if len(poly) >= 3 else []

    # Case 2: list of polygons of dict points
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], list) and len(obj[0]) > 0 and isinstance(obj[0][0], dict):
        polys: List[Polygon] = []
        for one in obj:
            poly = [(float(p["x"]), float(p["y"])) for p in one]
            if len(poly) >= 3:
                polys.append(poly)
        return polys

    # Case 3: list of [x,y] points => one polygon
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], (list, tuple)) and len(obj[0]) == 2 and isinstance(obj[0][0], (int, float)):
        poly = [(float(p[0]), float(p[1])) for p in obj]
        return [poly] if len(poly) >= 3 else []

    # Case 4: list of polygons, each is list of [x,y]
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], list) and len(obj[0]) > 0 and isinstance(obj[0][0], (list, tuple)):
        polys = []
        for one in obj:
            if not one:
                continue
            if isinstance(one[0], (list, tuple)) and len(one[0]) == 2:
                poly = [(float(p[0]), float(p[1])) for p in one]
                if len(poly) >= 3:
                    polys.append(poly)
        return polys

    return []


@dataclass
class VinDrRibConfig:
    ann_json: Path
    image_root: Path
    image_size: int = 224
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    mode: str = "multiclass"  # "multiclass" (0..K) or "binary" (0/1)


class VinDrRibSegDataset(Dataset):
    """
    VinDr-RibCXR segmentation dataset.

    Your annotation JSON looks like:
      {
        "img": {"0": "data/train/img/xxx.png", ...},
        "R1":  {"0": [ {x,y}, ... ], "1": [...], ...},
        "R2":  {...},
        ...
      }

    We rasterize polygons into a mask on-the-fly.
      - multiclass: background=0, rib_i = i (1..K) based on R<number> keys
      - binary: background=0, any rib=1

    Attributes you’ll use elsewhere:
      - self.class_names  (["background", "R1", "R2", ...])
      - self.num_classes  (1+K for multiclass, 2 for binary)
    """

    def __init__(
        self,
        ann_json: str | Path,
        image_root: str | Path,
        image_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        mode: str = "multiclass",
    ):
        super().__init__()
        cfg = VinDrRibConfig(
            ann_json=_as_path(ann_json),
            image_root=_as_path(image_root),
            image_size=int(image_size),
            mean=mean,
            std=std,
            mode=mode,
        )

        if cfg.mode not in ("multiclass", "binary"):
            raise ValueError(f"mode must be 'multiclass' or 'binary', got: {cfg.mode}")

        self.ann_json = cfg.ann_json
        self.image_root = cfg.image_root
        self.image_size = cfg.image_size
        self.mode = cfg.mode

        with open(self.ann_json, "r", encoding="utf-8") as f:
            self.ann: Dict[str, Any] = json.load(f)

        if "img" not in self.ann:
            raise ValueError(f"Annotation JSON missing top-level 'img' key: {self.ann_json}")

        # image index keys are strings: "0", "1", ...
        self.img_map: Dict[str, str] = self.ann["img"]
        self.indices: List[str] = sorted(self.img_map.keys(), key=_safe_int)

        # all rib keys: "R1", "R2", ...
        rib_keys = [k for k in self.ann.keys() if k != "img" and isinstance(k, str) and k.startswith("R")]
        # sort by numeric part
        rib_keys.sort(key=lambda k: _safe_int(k[1:], 0))
        self.rib_keys = rib_keys

        self.class_names = ["background"] + rib_keys
        if self.mode == "binary":
            self.num_classes = 2
        else:
            self.num_classes = 1 + len(rib_keys)

        self.transform_img = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ])

        # masks: resize with nearest (we’ll do via PIL in __getitem__)
        self._mask_resize = transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.NEAREST)

        # Precompute per-class per-index polygon blobs for speed (optional lightweight dict lookups)
        # ann["Rk"] is expected to be a dict: {"0": <poly>, "1": <poly>, ...}
        self._class_to_index_poly: Dict[str, Dict[str, Any]] = {}
        for rk in self.rib_keys:
            v = self.ann.get(rk, {})
            self._class_to_index_poly[rk] = v if isinstance(v, dict) else {}

    def __len__(self) -> int:
        return len(self.indices)

    def _resolve_image_path(self, rel_path: str) -> Path:
        """
        JSON paths look like: data/train/img/VinDr_RibCXR_train_000.png
        Disk paths look like: <root>/Data/train/img/VinDr_RibCXR_train_000.png

        We'll try a few sensible variants so you don't have to fight path prefixes.
        """
        rel = Path(rel_path)

        candidates: List[Path] = []

        # 1) root / rel as-is
        candidates.append(self.image_root / rel)

        # 2) swap leading "data" -> "Data"
        parts = list(rel.parts)
        if parts and parts[0].lower() == "data":
            parts2 = parts[:]
            parts2[0] = "Data"
            candidates.append(self.image_root / Path(*parts2))

        # 3) if user passed image_root that already points to .../Data
        #    then strip first component ("data" or "Data") and join
        if parts and parts[0].lower() in ("data", "Data".lower()):
            candidates.append(self.image_root / Path(*parts[1:]))

        # 4) absolute path inside JSON (rare)
        if rel.is_absolute():
            candidates.append(rel)

        for p in candidates:
            if p.exists():
                return p

        raise FileNotFoundError(
            f"Could not resolve image path.\n"
            f"  rel_path in JSON: {rel_path}\n"
            f"  image_root: {self.image_root}\n"
            f"  tried: {[str(c) for c in candidates]}"
        )

    def _rasterize_mask(self, img_w: int, img_h: int, idx_key: str) -> Image.Image:
        """
        Returns a PIL 'L' mask at original image resolution (img_w, img_h).
        """
        mask = Image.new("L", (img_w, img_h), 0)
        draw = ImageDraw.Draw(mask)

        if self.mode == "binary":
            fill_value = 1
            for rk in self.rib_keys:
                obj = self._class_to_index_poly.get(rk, {}).get(idx_key, None)
                for poly in _parse_polygon(obj):
                    draw.polygon(poly, fill=fill_value, outline=fill_value)
            return mask

        # multiclass: each rib key gets its own integer label (1..K)
        for class_id, rk in enumerate(self.rib_keys, start=1):
            obj = self._class_to_index_poly.get(rk, {}).get(idx_key, None)
            for poly in _parse_polygon(obj):
                draw.polygon(poly, fill=class_id, outline=class_id)

        return mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx_key = self.indices[idx]
        rel_img = self.img_map[idx_key]
        img_path = self._resolve_image_path(rel_img)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # build mask at original res, then resize w/ nearest
        mask_pil = self._rasterize_mask(w, h, idx_key)
        mask_pil = self._mask_resize(mask_pil)

        # image transform (resize + normalize)
        x = self.transform_img(img)

        # mask to tensor (long, shape [H,W])
        mask_np = np.array(mask_pil, dtype=np.int64)
        y = torch.from_numpy(mask_np).long()

        # If binary, ensure it's 0/1
        if self.mode == "binary":
            y = (y > 0).long()

        return x, y