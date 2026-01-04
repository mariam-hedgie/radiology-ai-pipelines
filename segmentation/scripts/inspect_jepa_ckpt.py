import torch
from pathlib import Path

CKPT_PATH = Path("/data1/mariam/anas_imp/jepa-latest.pth.tar")


def main():
    print("Loading:", CKPT_PATH)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    print("\nTop-level type:", type(ckpt))
    if not isinstance(ckpt, dict):
        print("Checkpoint is not a dict. Exiting.")
        return

    print("\nTop-level keys:")
    for k in ckpt.keys():
        print(" -", k)

    # Common nested candidates where the weights live
    candidates = ["state_dict", "model", "encoder", "student", "target", "ema", "net"]
    found = []
    for c in candidates:
        if c in ckpt:
            found.append(c)

    print("\nNested weight containers found:", found if found else "None of the common ones")

    # Choose the most likely container
    container = None
    for c in ["state_dict", "model", "encoder", "student", "ema", "target", "net"]:
        if c in ckpt and isinstance(ckpt[c], dict):
            container = ckpt[c]
            print(f"\nUsing ckpt['{c}'] as weight container (preview):")
            keys = list(container.keys())
            print("Num keys:", len(keys))
            print("First 40 keys:")
            for kk in keys[:40]:
                print("  ", kk)
            break

    # If weights are at top-level directly (rare but possible)
    if container is None:
        # Heuristic: top-level might already be a state_dict if keys look like module names
        tl_keys = list(ckpt.keys())
        if tl_keys and isinstance(ckpt[tl_keys[0]], torch.Tensor):
            print("\nTop-level looks like a raw state_dict (tensor values).")
            print("Num keys:", len(tl_keys))
            print("First 40 keys:")
            for kk in tl_keys[:40]:
                print("  ", kk)
        else:
            print("\nNo obvious state_dict container found yet.")

if __name__ == "__main__":
    main()