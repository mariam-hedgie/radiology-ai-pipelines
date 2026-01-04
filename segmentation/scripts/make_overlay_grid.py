import argparse
from pathlib import Path
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_img(p: Path):
    return mpimg.imread(str(p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch_dir", type=str, required=True,
                        help="Path like outputs/test/epoch_10")
    parser.add_argument("--n", type=int, default=24,
                        help="How many samples to include")
    parser.add_argument("--cols", type=int, default=4,
                        help="How many samples per row")
    parser.add_argument("--out", type=str, default=None,
                        help="Output file path (png). Default: <epoch_dir>/compare_grid.png")
    args = parser.parse_args()

    epoch_dir = Path(args.epoch_dir)
    assert epoch_dir.exists(), f"Not found: {epoch_dir}"

    # We’ll use the index prefix to pair files (0000_...)
    gt_files = sorted(epoch_dir.glob("*_gt_overlay.png"))
    pred_files = sorted(epoch_dir.glob("*_pred_overlay.png"))
    img_files = sorted(epoch_dir.glob("*_img.png"))

    # Build maps by sample id (prefix before first underscore)
    def sid(p: Path):
        return p.name.split("_")[0]

    gt_map = {sid(p): p for p in gt_files}
    pred_map = {sid(p): p for p in pred_files}
    img_map = {sid(p): p for p in img_files}

    common = sorted(set(gt_map) & set(pred_map) & set(img_map))
    if len(common) == 0:
        raise RuntimeError(
            "No matched samples found. Make sure you have *_img.png, *_gt_overlay.png, *_pred_overlay.png in the epoch folder."
        )

    n = min(args.n, len(common))
    common = common[:n]

    # Each sample gets 3 panels: IMG | GT overlay | PRED overlay
    panels_per_sample = 3
    total_panels = n * panels_per_sample

    # rows based on cols (#samples per row)
    samples_per_row = args.cols
    rows = math.ceil(n / samples_per_row)

    fig, axes = plt.subplots(
        rows,
        samples_per_row * panels_per_sample,
        figsize=(4.2 * samples_per_row, 3.2 * rows)
    )

    # axes might be 1D if rows==1
    if rows == 1:
        axes = [axes]

    for i, sample_id in enumerate(common):
        r = i // samples_per_row
        c0 = (i % samples_per_row) * panels_per_sample

        img = load_img(img_map[sample_id])
        gt = load_img(gt_map[sample_id])
        pred = load_img(pred_map[sample_id])

        for j, (arr, title) in enumerate([(img, "IMG"), (gt, "GT"), (pred, "PRED")]):
            ax = axes[r][c0 + j] if rows > 1 else axes[r][c0 + j]
            ax.imshow(arr)
            ax.set_title(f"{sample_id} {title}", fontsize=9)
            ax.axis("off")

    # Hide any unused axes
    for r in range(rows):
        for c in range(samples_per_row * panels_per_sample):
            ax = axes[r][c] if rows > 1 else axes[r][c]
            if not ax.has_data():
                ax.axis("off")

    fig.suptitle(f"Overlay comparison: {epoch_dir}", fontsize=14)
    plt.tight_layout()

    out_path = Path(args.out) if args.out else (epoch_dir / "compare_grid.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved grid to: {out_path}")


if __name__ == "__main__":
    main()