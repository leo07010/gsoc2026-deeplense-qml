# 03_Data — DeepLense datasets

The actual data (3+ GB, ~178k `.npy` files) is **not committed** to this repository.
Fetch it with the helper script from the repo root:

```bash
pip install gdown
python download_data.py        # downloads Dataset1.zip + Dataset2.zip into 03_Data/
```

Then unzip in place. Expected layout after extraction:

```
03_Data/
├── Dataset/
│   ├── axion/    *.npy   (~29,896 images)   # classification + MAE pretrain
│   ├── cdm/      *.npy   (~29,759 images)
│   └── no_sub/   *.npy   (~29,449 images)
├── HR/           *.npy   (~10,000 images)   # super-resolution (high-res)
└── LR/           *.npy   (~10,000 images)   # super-resolution (low-res)
```

## Sources

| File | Purpose | Google Drive |
|------|---------|--------------|
| `Dataset1.zip` (2.7 GB) | classification + MAE pretrain | [1znqUeFzYz-DeAE3dYXD17qoMPK82Whji](https://drive.google.com/file/d/1znqUeFzYz-DeAE3dYXD17qoMPK82Whji/view) |
| `Dataset2.zip` (509 MB) | super-resolution | [1uJmDZw649XS-r-dYs9WD-OPwF_TIroVw](https://drive.google.com/file/d/1uJmDZw649XS-r-dYs9WD-OPwF_TIroVw/view) |

Each `.npy` is a single-channel simulated strong-lensing image. The three
classification classes correspond to dark matter substructure types:
`axion` (vortex), `cdm` (cold dark matter subhalos), and `no_sub` (no substructure).
