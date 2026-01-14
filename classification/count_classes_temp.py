from pathlib import Path
import pandas as pd

# -------- CONFIG --------
ROOT = Path("/data1/mariam/anas_imp/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/annotations")

TRAIN_CSV = ROOT / "image_labels_train.csv"
TEST_CSV  = ROOT / "image_labels_test.csv"
# ------------------------


def count_split(csv_path: Path, split_name: str):
    df = pd.read_csv(csv_path)

    assert "image_id" in df.columns, "image_id column missing"

    label_cols = [c for c in df.columns if c != "image_id"]

    print(f"\n===== {split_name.upper()} SPLIT =====")
    print(f"Total images: {len(df)}\n")

    counts = {}
    for c in label_cols:
        # labels are 0/1
        counts[c] = int(df[c].sum())

    return counts, label_cols, len(df)


def main():
    train_counts, classes, n_train = count_split(TRAIN_CSV, "train")
    test_counts, _, n_test = count_split(TEST_CSV, "test")

    print("\n===== CLASS-WISE POSITIVE COUNTS =====")
    print(f"{'Class':30s} {'Train':>8s} {'Test':>8s}")
    print("-" * 50)

    for c in classes:
        print(f"{c:30s} {train_counts[c]:8d} {test_counts[c]:8d}")

    print("\n===== SUMMARY =====")
    print(f"Train images: {n_train}")
    print(f"Test  images: {n_test}")
    print(f"Num classes : {len(classes)}")


if __name__ == "__main__":
    main()