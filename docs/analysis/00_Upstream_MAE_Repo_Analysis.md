# mae-lensing — Repo 完整技術分析

> **對象**：[`achmadardanip/mae-lensing`](https://github.com/achmadardanip/mae-lensing)
> **分析人**：你的 GSoC workspace
> **分析日期**：2026-05-27
> **目的**：理解 repo 全貌，標定量子整合插入點，記錄已知 quirks

---

## 目錄

1. [Repo 身分與規模](#1-repo-身分與規模)
2. [檔案級分析](#2-檔案級分析)
3. [`mainv2.py` 程式碼架構](#3-mainv2py-程式碼架構)
4. [三階段 Pipeline 資料流](#4-三階段-pipeline-資料流)
5. [關鍵設計選擇與隱性假設](#5-關鍵設計選擇與隱性假設)
6. [預訓 weights 內容分析](#6-預訓-weights-內容分析)
7. [Output artifacts 完整清單](#7-output-artifacts-完整清單)
8. [量子整合插入點（line-level）](#8-量子整合插入點line-level)
9. [已知 quirks 與 gotchas](#9-已知-quirks-與-gotchas)
10. [改動建議：哪些可改、哪些不要動](#10-改動建議哪些可改哪些不要動)
11. [Reproduction guide](#11-reproduction-guide)
12. [整合 checklist](#12-整合-checklist)

---

## 1. Repo 身分與規模

| 項目 | 內容 |
|---|---|
| **論文** | *Masked Autoencoder Pretraining on Strong-Lensing Images for Joint Dark-Matter Model Classification and Super-Resolution* |
| **arXiv** | [2512.06642](https://arxiv.org/abs/2512.06642) (Dec 2025) |
| **作者** (8 人) | Achmad Ardani Prasha, Clavino Ourizqi Rachmadi, Muhamad Fauzan Ibnu Syahlan, Naufal Rahfi Anugerah, Nanda Garin Raditya, Putri Amelia, Sabrina Laila Mutiara, Hilman Syachr Ramadhan |
| **單位** | Universitas Mercu Buana, Jakarta, Indonesia |
| **GitHub** | <https://github.com/achmadardanip/mae-lensing> |
| **License** | repo 內未明確標示（建議 fork 前 email 確認）|
| **Repo 規模** | 1 主腳本（1833 行）+ 1 分析腳本（148 行）+ 1 notebook + 預訓 weights |
| **語言** | Python (PyTorch) |
| **目的** | Reproduce paper 結果；不是 production-grade library |

---

## 2. 檔案級分析

### 2.1 全部檔案清單

```
mae-lensing/
├── .git/                                                    # version control
├── README.md                                          5.2 KB # 使用說明
├── mainv2.py                                         66.4 KB # 唯一主程式 (1833 行)
├── analyze_ablation.py                                5.8 KB # CSV → 圖
├── achmads-notebook-dec-5 (1).ipynb               16,629 KB # 開發 notebook (含原始輸出)
├── Masked_Autoencoder_Pretraining_..._(2).pdf       543 KB # paper PDF
└── outputs_lens/                                            # 範例 artifacts
    ├── classifier.pth                            10,921 KB # 預訓 classifier 權重 ⭐
    ├── sr_model.pth                              11,752 KB # 預訓 SR 模型權重 ⭐
    ├── ablation_cls.csv                              262 B # 3 行 ablation 結果
    ├── ablation_sr.csv                               171 B # 2 行 SR ablation
    ├── mask_ratio_ablation.csv                       515 B # 3 個 mask ratio 結果
    ├── cls_pretrained_classification_report.txt      389 B # per-class P/R/F1
    ├── cls_pretrained_confusion_matrix.png          27 KB
    ├── ablation_cls_summary.png                     27 KB
    ├── ablation_sr_summary.png                      27 KB
    ├── mask_ratio_ablation_summary.png              45 KB
    ├── reliability_pretrained.png                   38 KB
    ├── roc_curve.png                                44 KB
    ├── sr_example.png                               29 KB
    ├── sr_grid.png                                  143 KB
    └── tsne.png                                     84 KB
```

### 2.2 觀察與評論

| 觀察 | 含意 |
|---|---|
| **`mainv2.py` 是唯一程式碼** | 沒 module 化（沒 `models/`, `data/`, `train.py`），改一處要找全 1833 行 |
| **`outputs_lens` 已含預訓 weights** | README 沒明說但確實 ship 了——你**不需要重訓 2 小時 MAE** |
| **沒 `mae_encoder.pth` 獨立檔** | 只有 fine-tuned 整模型（含 encoder + head）|
| **`achmads-notebook` 16 MB 超大** | 內含 cell outputs（train log + 圖）；不適合進 git，但保留作為 reference |
| **沒 `requirements.txt`** | 環境靠 README pip 命令重建，pinned 版本未指定 |
| **沒 unit tests** | 純 research code，重構時要小心 |
| **沒 LICENSE 檔** | 法律不明確；要做 derivative work 前最好聯絡作者 |

---

## 3. `mainv2.py` 程式碼架構

### 3.1 全圖（1833 行的 5 大區塊）

```
Line 1-38       Imports + try/except for timm
Line 40-194     Utils & data loading (seed, zip extraction, robust npy loader)
                 ├─ set_seed(), extract_zip_if_exists(), find_dataset_root_for_classes()
                 ├─ _extract_2d_numeric_array() ← 處理「奇怪格式 npy」的 18-行 fallback
                 └─ robust_load_npy() ← 含 noise fallback（重要 quirk!）

Line 197-343    Datasets (3 個 Dataset class + 2 個 builder)
                 ├─ NoSubDataset (line 201) ← MAE pretrain 用，只用 no_sub 類！
                 ├─ ClassificationDataset (line 217) ← 3-way
                 ├─ SuperResolutionDataset (line 237)
                 ├─ build_task_VI_A_datasets() (line 256)
                 └─ build_task_VI_B_datasets() (line 307)

Line 346-609    Models (8 個 nn.Module class)
                 ├─ PatchEmbed (line 350)
                 ├─ MLP (line 375)
                 ├─ TransformerBlock (line 394)
                 ├─ ViTEncoder (line 421) ⭐ 你要凍結這個
                 ├─ MaskedAutoencoderViT (line 484) ⭐ pretrain 模型
                 ├─ ViTClassifier (line 554) ⭐ 量子插入點！
                 ├─ SRHead (line 567)
                 └─ ViTSuperResolution (line 596)

Line 612-977    Training / Eval / Plotting (15 個函式)
                 ├─ train_mae() (line 616)
                 ├─ evaluate_classifier() (line 648)
                 ├─ plot_roc_curves(), plot_tsne()
                 ├─ evaluate_sr(), visualize_sr_example(), visualize_sr_grid()
                 ├─ plot_confusion_and_report()
                 └─ plot_reliability_diagram()

Line 980-1102   Optuna objectives (2 個)
                 ├─ objective_classification() ← 5 trials，搜 lr/wd/drop
                 └─ objective_sr()

Line 1105-1493  Experiment runners (4 個)
                 ├─ classifier_experiment() ← scratch vs pretrained vs frozen
                 ├─ sr_experiment() ← scratch vs pretrained
                 ├─ run_ablation_experiments() ← 跑上面兩個
                 └─ mask_ratio_experiment() + run_mask_ratio_experiments()

Line 1495-1833  main() ← 唯一 entry point，硬編三階段流程
```

### 3.2 8 個模型 class 詳解

#### `PatchEmbed` (line 350)
```python
img 64×64 → Conv2d(1, 192, k=4, s=4) → (B, 192, 16, 16)
         → flatten + transpose → (B, 256, 192)  # 256 tokens
```

#### `TransformerBlock` (line 394)
標準 ViT block：
```
x + MHA(LayerNorm(x)) → + MLP(LayerNorm(x))
```

#### `ViTEncoder` (line 421) ⭐
```
img 64×64
  → PatchEmbed → (B, 256, 192)
  → concat CLS token → (B, 257, 192)
  → + pos_embed
  → 6 × TransformerBlock
  → LayerNorm
  → 輸出 (B, 257, 192)
```
- **參數量約 1.0 M**
- 提供兩個 forward：`forward()`（從 image）與 `forward_features_from_patches()`（從 patches，MAE 用）

#### `MaskedAutoencoderViT` (line 484)
完整 MAE：
```
img → patches → random_mask (90% zero) → encoder → decoder_embed (linear 192→192)
   → 2 × TransformerBlock (decoder) → linear 192 → 4·4·1 = 16
   → unpatchify → reconstructed image
```
- **`random_mask` 把 masked patch 設為 0** （不是真正的「移除」——這是 SimMIM-style，非 He et al. MAE）
- Decoder 只有 **2 blocks**，比 encoder（6 blocks）小

#### `ViTClassifier` (line 554) ⭐⭐ 量子插入點
**只有 10 行**：
```python
class ViTClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, num_classes)  # ← 替換這行
    def forward(self, x):
        tokens = self.encoder(x)
        cls_feat = tokens[:, 0]
        logits = self.head(cls_feat)
        return logits, cls_feat
```

#### `ViTSuperResolution` (line 596)
```
img 16×16
  → encoder → patch tokens (B, 16, 192)
  → reshape (B, 192, 4, 4)
  → SRHead: 2× Upsample + Conv → 16×16 → 64×64 → final 1-ch
```

### 3.3 評估函式 (`evaluate_classifier`, line 648)

回傳 dict：
```python
{
  "auc_macro", "fpr", "tpr",
  "probs", "labels", "features",   # ← features 可用 t-SNE 視覺化
  "accuracy", "f1_macro", "y_pred",
}
```

---

## 4. 三階段 Pipeline 資料流

### 4.1 完整時序

```
═══════════════════════════════════════════════════════════════
Phase 1: MAE Pretraining (no_sub class only!)
═══════════════════════════════════════════════════════════════

Dataset1.zip
  ├─ axion/   (29,896 .npy) ──────────── 訓 CLS 用
  ├─ cdm/     (29,759 .npy) ──────────── 訓 CLS 用
  └─ no_sub/  (29,449 .npy) ──────┐
                                   │ ⭐ pretrain MAE 只用這
  no_sub paths ──→ NoSubDataset  ──┘
                ↓
       MaskedAutoencoderViT (mask_ratio=0.75 默認, 但 0.9 才是 SOTA)
                ↓
       train_mae() → 10 epochs Adam, lr=1e-4
                ↓
       mae_encoder.pth saved

═══════════════════════════════════════════════════════════════
Phase 2: Classification Fine-tuning (all 3 classes)
═══════════════════════════════════════════════════════════════

Dataset1.zip (axion + cdm + no_sub all)
  → build_task_VI_A_datasets() → train_cls_ds (90%) / val_cls_ds (10%)
                ↓
       new ViTEncoder ← load mae_encoder.pth weights
                ↓
       ViTClassifier(encoder, num_classes=3)
                ↓
       10 epochs Adam, lr=5e-5
                ↓
       evaluate_classifier() → AUC, ACC, F1, ROC, t-SNE, confusion
                ↓
       classifier.pth saved (含 encoder + head 一起)

═══════════════════════════════════════════════════════════════
Phase 3: Super-Resolution Fine-tuning
═══════════════════════════════════════════════════════════════

Dataset2.zip
  ├─ HR/  (10,000 .npy)
  └─ LR/  (10,000 .npy)
       → SuperResolutionDataset (按檔名 match)
                ↓
       new ViTEncoder ← load mae_encoder.pth
                ↓
       ViTSuperResolution(encoder)
                ↓
       10 epochs MSE
                ↓
       PSNR, SSIM 評估
                ↓
       sr_model.pth saved
```

### 4.2 三階段共用元件

| 元件 | Phase 1 用 | Phase 2 用 | Phase 3 用 |
|---|---|---|---|
| `ViTEncoder` | ✓ (新 init) | ✓ (load MAE weights) | ✓ (load MAE weights) |
| 影像尺寸 | 64×64 hard-coded | 64×64 | 64×64 |
| Batch size | 64 | 64 | 64 |
| Optimizer | Adam | Adam | Adam |
| Loss | MSE | CrossEntropy | MSE |
| Epochs | 10 | 10 | 10 |

---

## 5. 關鍵設計選擇與隱性假設

### 5.1 ⭐ MAE 只在 `no_sub` 上預訓（強假設！）

```python
# Line 280-281
if cls == "no_sub":
    no_sub_paths.append(fpath)

# Line 302
mae_ds = NoSubDataset(no_sub_paths, target_size=target_size)
```

**為什麼這是強假設**：
- 標準 MAE 在所有資料上做 SSL（self-supervised，不需要 label）
- 但這個 repo 把 MAE 限制只看「無 substructure」的影像
- **物理直覺**：讓 encoder 學「乾淨 arc」長什麼樣 → 後續分類時，subhalo/vortex 會被視為「對 baseline 的擾動」
- **副作用**：encoder 對 substructure 的 representation 沒有 unsupervised exposure，可能限制下游 axion 分類能力（這呼應你看到的 axion recall 0.76 偏低）

> **你的量子方向可以挑戰這假設**：用全部 3 類做 MAE pretrain，看 quantum head 是否能利用更豐富的 representation

### 5.2 ⭐ 影像強制 resize 到 64×64

```python
# Line 183 (robust_load_npy)
img = F.interpolate(img, size=(target_size, target_size), mode="bicubic", ...)
```

- 原始 lensing 影像可能是 150×150（Alexander 2019 用的尺寸），被縮成 64×64
- **資訊損失**：subhalo 在原圖約 1-2 pixel 大小，縮 ×2.3 後可能消失
- **改動風險**：若你想用 150×150，需同時改 `target_size=150` + `patch_size=10` （讓 num_patches 保持合理）+ `pos_embed` shape

### 5.3 ⭐ Mask token 是「設 0」而非真移除（SimMIM 風格）

```python
# Line 514-525 (random_mask)
x_masked = x_patches.clone()
x_masked[mask] = 0.0
return x_masked, mask
```

- He et al. 原版 MAE 是把 masked patches **從 encoder input 移除**，encoder 只看 visible（這節省計算）
- 這 repo 是 **set to zero**，encoder 仍看 256 個 tokens（其中 90% 是 zero）
- **效能差異**：He 版較快，這版較慢但實作簡單

### 5.4 ⭐ Decoder 比 He 原版更小（2 vs 8 blocks）

| 設計 | He 2022 | 這 repo |
|---|---|---|
| Encoder depth | 24 (ViT-Huge) | **6** |
| Decoder depth | 8 | **2** |
| Encoder dim | 1280 | **192** |
| Decoder dim | 512 | **192** |

→ 這 repo 大幅縮減模型，**約 ViT-Tiny 量級**（總 ~1.5M params），適合單卡訓練。

### 5.5 沒有獨立 test set

- `val_fraction=0.1` 拆 train/val
- **沒有 held-out test set**——所有「test」其實是 val
- **後果**：報告數字理論上有 selection bias（在 val 上 tune 後又在 val 上報告）

### 5.6 Random_mask 是 per-sample 而非 per-batch

```python
# Line 515-521
for i in range(B):  # ← 每個 sample 各別 mask
    perm = torch.randperm(N, device=x_patches.device)
    ...
```

- 每張影像的 mask pattern 都不同
- 比 per-batch mask 提供更多隨機性，但慢一點（Python loop over batch）

### 5.7 沒用 He 原版的 normalized pixel loss

He 2022 推薦對每個 patch 的 pixel 做 LayerNorm 後再算 MSE。這 repo 直接對 raw pixel 算 MSE：
```python
# Line 626
criterion = nn.MSELoss()
# Line 637
loss = criterion(recon, imgs)
```

→ 可能讓收斂稍慢、final loss 偏小。

---

## 6. 預訓 weights 內容分析

### `outputs_lens/classifier.pth` (10.9 MB)

由 `torch.save(classifier.state_dict(), ...)` 儲存（line 1698-1699）。預期 state_dict keys：

```python
{
    'encoder.cls_token': (1, 1, 192),
    'encoder.pos_embed': (1, 257, 192),
    'encoder.patch_embed.proj.weight': (192, 1, 4, 4),
    'encoder.patch_embed.proj.bias': (192,),
    'encoder.blocks.0.norm1.weight/bias': (192,) ×2,
    'encoder.blocks.0.attn.in_proj_weight': (576, 192),
    'encoder.blocks.0.attn.in_proj_bias': (576,),
    'encoder.blocks.0.attn.out_proj.weight': (192, 192),
    'encoder.blocks.0.attn.out_proj.bias': (192,),
    'encoder.blocks.0.norm2.weight/bias': (192,) ×2,
    'encoder.blocks.0.mlp.fc1.weight': (768, 192),
    'encoder.blocks.0.mlp.fc1.bias': (768,),
    'encoder.blocks.0.mlp.fc2.weight': (192, 768),
    'encoder.blocks.0.mlp.fc2.bias': (192,),
    # ... blocks 1-5 重複 ...
    'encoder.norm.weight/bias': (192,) ×2,
    'head.weight': (3, 192),
    'head.bias': (3,),
}
```

### 抽出 encoder weights 給量子 fine-tune 用

```python
import torch

ckpt = torch.load('outputs_lens/classifier.pth', map_location='cpu')
encoder_state = {k.replace('encoder.', ''): v
                 for k, v in ckpt.items()
                 if k.startswith('encoder.')}

# 載入到新 encoder
from mainv2 import ViTEncoder
enc = ViTEncoder(img_size=64, patch_size=4, in_chans=1,
                 embed_dim=192, depth=6, num_heads=3)
enc.load_state_dict(encoder_state)
enc.eval()
for p in enc.parameters(): p.requires_grad = False
# 接你的 QuantumFusionHead
```

---

## 7. Output artifacts 完整清單

| 檔案 | 內容 | 對你的價值 |
|---|---|---|
| `classifier.pth` | encoder + head 全 model | **直接抽 encoder 用** |
| `sr_model.pth` | encoder + SR head | 視情況可用 |
| `ablation_cls.csv` | 3 行：pretrained_full / pretrained_frozen / scratch_full | 確認 pretrain 在 5 epochs ablation 中沒贏 scratch |
| `ablation_sr.csv` | 2 行：pretrained_sr / scratch_sr | SR 也是 pretrain 小贏 |
| `mask_ratio_ablation.csv` | 3 行 mask 0.5/0.75/0.9 | 確認 90% 才到 AUC 0.968 |
| `cls_pretrained_classification_report.txt` | per-class P/R/F1 | **axion recall 0.76** 是改進空間 |
| `roc_curve.png` | 3 個 class 的 ROC | 視覺化比對基準 |
| `tsne.png` | t-SNE of encoder features | 看 representation 是否分離 |
| `confusion_matrix.png` | 3×3 混淆矩陣 | axion ↔ cdm 混淆嚴重 |
| `reliability_pretrained.png` | calibration plot | 看 model 自信度是否準確 |
| `sr_example.png` / `sr_grid.png` | LR / Pred / HR 並列 | SR 視覺品質 |

---

## 8. 量子整合插入點（line-level）

### 🎯 主要插入點：`ViTClassifier` (line 554-564)

**原始 30 行**：
```python
class ViTClassifier(nn.Module):
    def __init__(self, encoder: ViTEncoder, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, num_classes)  # ⬅ 替換這個

    def forward(self, x: torch.Tensor):
        tokens = self.encoder(x)
        cls_feat = tokens[:, 0]
        logits = self.head(cls_feat)                            # ⬅ 用 fusion module
        return logits, cls_feat
```

**改成**：
```python
from quantum_fusion import QuantumFusionHead   # 你新加的 module

class ViTClassifier(nn.Module):
    def __init__(self, encoder: ViTEncoder, num_classes: int,
                 use_quantum: bool = False):                    # ⬅ 加旗標
        super().__init__()
        self.encoder = encoder
        if use_quantum:
            self.head = QuantumFusionHead(in_dim=encoder.embed_dim,
                                          n_classes=num_classes)
        else:
            self.head = nn.Linear(encoder.embed_dim, num_classes)

    def forward(self, x: torch.Tensor):
        tokens = self.encoder(x)
        cls_feat = tokens[:, 0]
        logits, *_ = self.head(cls_feat) \
                     if isinstance(self.head, QuantumFusionHead) \
                     else (self.head(cls_feat),)
        return logits, cls_feat
```

### 🎯 CLI 旗標：`main()` (line 1495)

加一個 `--use_quantum` argparse：
```python
parser.add_argument("--use_quantum", action="store_true",
                    help="Use QuantumFusionHead instead of Linear head.")
```

並在 line 1641 `ViTClassifier` 初始化處：
```python
classifier = ViTClassifier(encoder_cls,
                           num_classes=len(class_names),
                           use_quantum=args.use_quantum).to(device)
```

### 🎯 次要插入點

| 位置 | line | 改動類型 |
|---|---|---|
| `ViTSuperResolution` (line 596) | — | 若做 quantum SR head，類似 ViTClassifier |
| `MaskedAutoencoderViT` decoder (line 502-512) | — | 若做 Q-MAE，把 decoder 換成 quantum |
| `random_mask()` (line 514) | — | 若做 quantum mask strategy，這裡改 |
| `train_mae()` (line 616) | — | 若加 quantum reconstruction loss，這裡改 |

### 🎯 建議的新檔案結構（量子整合後）

```
mae-lensing/
├── mainv2.py                      # 原 paper code（盡量少改）
├── quantum_fusion.py              # 你新增：QuantumFusionHead class
├── train_quantum.py               # 你新增：呼叫 mainv2 + quantum 模式
└── outputs_quantum/               # 量子實驗結果
    ├── q_classifier.pth
    └── ablation_q_vs_c.csv
```

---

## 9. 已知 quirks 與 gotchas

### 9.1 `robust_load_npy` 失敗時默默用噪音（line 191-194）

```python
except Exception as e:
    print(f"[WARN] Failed to load {path}: {e}. Using random noise.")
    noise = torch.rand(1, target_size, target_size)
    return noise
```

→ 若資料集有壞檔，**訓練不會崩**但會被 noise 污染。**最好 grep `[WARN] Failed to load`** 看 log。

### 9.2 `os.chdir(args.data_root)` (line 1540)

```python
os.chdir(args.data_root)
os.makedirs("outputs", exist_ok=True)
```

→ 改變了 cwd，且 `outputs/` 是在 `data_root/` 而不是 repo 根。**寫量子 import 路徑時要小心**（用絕對路徑或在 `os.chdir` 前 import）。

### 9.3 默認 mask_ratio 0.75，但 SOTA 是 0.9

```python
# Line 1505
parser.add_argument("--mae_mask_ratio", type=float, default=0.75, ...)
```

要重現 SOTA AUC 0.968，**必須加 `--mae_mask_ratio 0.9`**。

### 9.4 Optuna 只跑 5 trials

```python
# Line 1510-1511
parser.add_argument("--optuna_trials_cls", type=int, default=5)
parser.add_argument("--optuna_trials_sr", type=int, default=5)
```

5 trials 對 3-d hyperparameter space (lr, wd, drop) 太少。**正經實驗應 >= 20 trials**。

### 9.5 No reproducibility on CUDA

```python
# Line 49-50
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

設了 deterministic 但 PyTorch 在 CUDA 上某些 op 仍非 deterministic（如 atomic add）。**完全重現 seed 在 CUDA 下不保證**。

### 9.6 Random_mask 是 per-sample Python loop（慢）

```python
# Line 518-521
for i in range(B):
    perm = torch.randperm(N, device=x_patches.device)
    ...
```

可以向量化但作者沒做。**對 batch 64 影響不大但 batch 256+ 會明顯慢**。

### 9.7 ViTEncoder 在 main() 中被 instantiated 4 次

Line 1579, 1632, 1727, +ablation 多次。**每次都重新初始化** weights 再 load_state_dict。改 architecture 時要記得 4 處都改。

---

## 10. 改動建議：哪些可改、哪些不要動

### ✅ 安全可改

| 改動 | 影響範圍 |
|---|---|
| 加 `QuantumFusionHead` class | 新檔，不動 mainv2.py 即可（用 monkey-patch）|
| 加 `--use_quantum` argparse | 只動 `main()` 與 `ViTClassifier` |
| 改 `batch_size` | 純 CLI 參數 |
| 加新的 evaluation metrics（ECE, Brier）| 只動 `evaluate_classifier` |

### ⚠️ 可改但要小心

| 改動 | 連帶影響 |
|---|---|
| 改 `img_size` 從 64 → 150 | 需同步改 patch_size、num_patches、pos_embed shape、interpolate target |
| 改 `embed_dim` 從 192 → 其他 | 影響所有 down/up 投影、head 維度 |
| 改 `depth`（encoder blocks）| 改完 mae_encoder.pth 不能 load（dim mismatch）|
| 改 MAE 預訓資料（加入 axion/cdm）| 違反作者「pretrain only no_sub」設計，但可能更好 |

### 🚫 不建議改

| 改動 | 為什麼 |
|---|---|
| `robust_load_npy` 的 noise fallback | 移掉後壞檔會 crash 整個訓練 |
| `os.chdir` 的位置 | 改了會破壞所有 output 路徑 |
| Per-sample masking | 雖然慢但 paper 是這樣訓的，改了 reproduce 不到 |

---

## 11. Reproduction guide

### Step 1：環境（建議用 conda env）

```bash
conda create -n maelens python=3.10 -y
conda activate maelens
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy scikit-learn scikit-image matplotlib tqdm optuna
```

### Step 2：下載資料

```bash
mkdir data && cd data
# 手動從 Google Drive 下載
# Dataset1.zip: https://drive.google.com/file/d/1znqUeFzYz-DeAE3dYXD17qoMPK82Whji/
# Dataset2.zip: https://drive.google.com/file/d/1uJmDZw649XS-r-dYs9WD-OPwF_TIroVw/
```

或用 `gdown`：
```bash
pip install gdown
gdown --id 1znqUeFzYz-DeAE3dYXD17qoMPK82Whji -O Dataset1.zip
gdown --id 1uJmDZw649XS-r-dYs9WD-OPwF_TIroVw -O Dataset2.zip
```

### Step 3：跑 baseline 重現 paper

```bash
cd ..  # 回到 mae-lensing/
python mainv2.py --data_root ./data --mae_mask_ratio 0.9
# 預期：~2-3 小時 on A100，輸出在 data/outputs/
```

### Step 4：跑 ablation

```bash
python mainv2.py --data_root ./data --mae_mask_ratio 0.9 --run_ablation \
                 --run_mask_ablation --mask_ratios 0.5,0.75,0.9
```

### Step 5：對照 paper 數字

| Metric | Paper | 你應該得到 |
|---|---|---|
| Macro AUC (mask 0.9) | 0.968 | 0.965 ± 0.005 |
| Acc (mask 0.9) | 88.65% | 87-89% |
| Per-class F1 axion | 0.854 | ~0.85 |

---

## 12. 整合 checklist

### Pre-flight（量子整合前要確認）

- [ ] 跑 baseline 並重現 paper AUC 0.968 ± 0.01
- [ ] 確認 `outputs_lens/classifier.pth` 能成功 `torch.load`
- [ ] 抽出 encoder weights 存成獨立 `mae_encoder_only.pth`
- [ ] 確認 PennyLane 環境裝好：`python -c "import pennylane; print(pennylane.__version__)"`

### 整合步驟

- [ ] 新增 `quantum_fusion.py` 含 `QuantumFusionHead` class
- [ ] 修改 `mainv2.py:554` 的 `ViTClassifier` 接受 `use_quantum` 旗標
- [ ] 加 `--use_quantum` CLI 旗標到 `main()`
- [ ] 寫 `train_quantum.py` 把 frozen MAE encoder + quantum head fine-tune
- [ ] 跑 baseline 對比：classical head vs quantum head（同 epoch 數）
- [ ] 跑 4-way ablation：
  - [ ] no fusion (linear head only)
  - [ ] quantum only (no classical residual)
  - [ ] concat fusion
  - [ ] cross-attn + TSHF
- [ ] 把結果寫進 CSV，套 `analyze_ablation.py` 的格式

### Post-flight（GSoC final report）

- [ ] Update 04_GSoC_QML_Proposal.md 加入實驗結果
- [ ] PR 給 ML4SCI/DeepLense 主 repo（不是 mae-lensing fork）
- [ ] Email 給 8 個原作者報告 derivative work
- [ ] 整理進 short paper 投稿 NeurIPS 2026 ML4PS workshop

---

## 13. 一句話總結

> **mae-lensing 是 paper-quality reproduction code**，**結構簡單（單檔 1833 行）但設計選擇有 quirks**（只用 no_sub 預訓、64×64 強制縮放、SimMIM-style masking 而非 He MAE-style）。
> **量子整合的最佳插入點是第 554 行的 `ViTClassifier`**，**只需替換 `nn.Linear` 為 `QuantumFusionHead`** 即可，**其餘 1800 多行可不動**。
> **預訓 weights 已 ship**（`outputs_lens/classifier.pth`），所以**你不需要 2 小時 MAE 重訓**——直接抽 encoder 接量子 head fine-tune 即可。

---

## 附錄：用一張圖看完整 repo

```
┌─────────────────────────────────────────────────────────────┐
│                    mae-lensing repo                         │
│                                                             │
│  ┌────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │ Dataset1   │───▶│  PHASE 1 (MAE)  │───▶│ encoder.pth │  │
│  │ (no_sub)   │    │   10 epochs     │    └──────┬──────┘  │
│  └────────────┘    └─────────────────┘           │         │
│                                                   │         │
│  ┌────────────┐    ┌─────────────────┐           │         │
│  │ Dataset1   │───▶│ PHASE 2 (CLS)   │◀──────────┤         │
│  │ (3 class)  │    │   10 epochs     │           │         │
│  └────────────┘    └────────┬────────┘           │         │
│                              │                   │         │
│                              ▼                   │         │
│                       ┌──────────────┐           │         │
│                       │classifier.pth│ ⭐ ship'd │         │
│                       └──────────────┘           │         │
│                                                   │         │
│  ┌────────────┐    ┌─────────────────┐           │         │
│  │ Dataset2   │───▶│ PHASE 3 (SR)    │◀──────────┘         │
│  │ (LR/HR)    │    │   10 epochs     │                     │
│  └────────────┘    └────────┬────────┘                     │
│                              │                              │
│                              ▼                              │
│                       ┌──────────────┐                     │
│                       │ sr_model.pth │ ⭐ ship'd            │
│                       └──────────────┘                     │
│                                                             │
│  量子整合插入點 ──▶  line 554 (ViTClassifier)               │
└─────────────────────────────────────────────────────────────┘
```

---

> **VSCode 預覽**：`Ctrl+Shift+V`
> **整合到主 README**：可在頂層 [README.md](../../README.md) 加 link 指向此檔
