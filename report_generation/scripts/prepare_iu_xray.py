# scripts/prepare_iu_xray.py
import csv
import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

ROOT = Path("data/iu_xray")
REPORTS_CSV = ROOT / "indiana_reports.csv"
PROJ_CSV = ROOT / "indiana_projections.csv"
IMAGES_DIR = ROOT / "images" / "images_normalized"

OUT_TRAIN = ROOT / "train.jsonl"
OUT_VAL = ROOT / "val.jsonl"
OUT_TEST = ROOT / "test.jsonl"


def read_reports():
    reports = {}
    with open(REPORTS_CSV, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            uid = row["uid"].strip()
            findings = (row.get("findings") or "").strip()
            impression = (row.get("impression") or "").strip()
            if not findings and not impression:
                continue
            reports[uid] = {"findings": findings, "impression": impression}
    return reports


def read_frontal_images():
    img_map = defaultdict(list)
    with open(PROJ_CSV, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            uid = row["uid"].strip()
            fn = row["filename"].strip()
            proj = (row["projection"] or "").strip().lower()
            if proj == "frontal":
                img_map[uid].append(fn)
    return img_map


def make_records(reports, img_map):
    records = []
    missing_img = 0
    missing_rep = 0

    for uid, fns in img_map.items():
        if uid not in reports:
            missing_rep += 1
            continue

        rep = reports[uid]
        # You can do impression-only if you want: text = rep["impression"]
        text = f"Findings: {rep['findings']} Impression: {rep['impression']}".strip()

        for fn in fns:
            img_path = IMAGES_DIR / fn
            if not img_path.exists():
                missing_img += 1
                continue

            records.append({
                "image": f"images/images_normalized/{fn}",
                "text": text,
                "uid": uid
            })

    print(f"records: {len(records)}")
    print(f"missing reports for uid: {missing_rep}")
    print(f"missing images on disk: {missing_img}")
    return records


def split(records, train_frac=0.8, val_frac=0.1):
    random.shuffle(records)
    n = len(records)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]
    return train, val, test


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} -> {path}")


def main():
    assert REPORTS_CSV.exists(), f"Missing {REPORTS_CSV}"
    assert PROJ_CSV.exists(), f"Missing {PROJ_CSV}"
    assert IMAGES_DIR.exists(), f"Missing {IMAGES_DIR}"

    reports = read_reports()
    img_map = read_frontal_images()
    records = make_records(reports, img_map)

    assert len(records) > 0, "No records created. Something is wrong with joins or paths."

    train, val, test = split(records)

    write_jsonl(OUT_TRAIN, train)
    write_jsonl(OUT_VAL, val)
    write_jsonl(OUT_TEST, test)


if __name__ == "__main__":
    main()