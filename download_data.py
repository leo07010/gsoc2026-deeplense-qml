#!/usr/bin/env python
"""
Download DeepLense Task VI A and B datasets to 03_Data/.
Falls back to printing manual download URLs if gdown is rate-limited.

Usage:
    cd C:\\Users\\USER\\Downloads\\GSoC
    python download_data.py
"""
import os
import sys
import subprocess


DATASETS = {
    "Dataset1.zip": "1znqUeFzYz-DeAE3dYXD17qoMPK82Whji",  # classification + MAE
    "Dataset2.zip": "1uJmDZw649XS-r-dYs9WD-OPwF_TIroVw",  # super-resolution
}

TARGET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_Data")


def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"[INFO] Target folder: {TARGET_DIR}")

    try:
        import gdown
    except ImportError:
        print("[ERROR] gdown not installed. Run: pip install gdown")
        sys.exit(1)

    for fname, fid in DATASETS.items():
        out = os.path.join(TARGET_DIR, fname)
        if os.path.exists(out) and os.path.getsize(out) > 1_000_000:
            print(f"[SKIP] {fname} already exists ({os.path.getsize(out)//(1024*1024)} MB)")
            continue

        print(f"\n[DOWNLOAD] {fname} (Google Drive ID: {fid})")
        try:
            gdown.download(id=fid, output=out, quiet=False)
        except Exception as e:
            print(f"[FAIL] gdown failed for {fname}: {e}")
            print(f"       Manual download:")
            print(f"       https://drive.google.com/file/d/{fid}/view")
            print(f"       Save as: {out}")
            continue

        if os.path.exists(out):
            mb = os.path.getsize(out) / (1024 * 1024)
            print(f"[OK] Downloaded {fname}: {mb:.1f} MB")

    print("\n[DONE] Now run:")
    print(f"  cd 02_Code\\mae-lensing")
    print(f"  python eval_pretrained.py --data_root ..\\..\\03_Data")


if __name__ == "__main__":
    main()
