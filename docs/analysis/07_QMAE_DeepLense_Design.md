# DeepLense-QMAE — Quantum Masked Autoencoder for Strong-Lensing Dark Matter

> **狀態**：設計 + 原型中
> **基礎論文**：Andrews et al., *Quantum Masked Autoencoders for Vision Learning*,
> [arXiv:2511.17372](https://arxiv.org/abs/2511.17372)(2025-11)
> **應用**：把 QMAE 第一次套到 DeepLense 強引力透鏡 dark matter 影像(axion / cdm / no_sub)
> **後端**：PennyLane `default.qubit` + backprop（與專案其餘 QML 同一條 SLURM/PennyLane pipeline）

---

## 1. 動機與定位

判別線(CNN→ViT→MAE)的古典 baseline 已達 **AUC 0.968**，我們的多組實驗(gated / xattn / QCT + sham 對照)顯示：
**在強古典 baseline + 海量資料下，量子在判別任務上沒有可測量的增益**（QCT quantum = QCT sham）。

QMAE 改走 **生成 / 自監督重建** 路線：
- 量子相對有理論優勢的是「表達分布 / 重建」，而非判別。
- DeepLense 過去的生成線(Alexander 2021 的 VAE/AAE)是**二分異常偵測**，**沒有人在三分類或在 MAE 框架下做過量子版** → 真 novelty。

**誠實定位**：這是 **proof-of-concept**，對標**量子 baseline(QAE)+ 同容量古典 sham**，
**不是**要贏古典 MAE 的 0.968。QMAE 原文在 MNIST 也只有 65%（量子尺度的玩具表現）。

---

## 2. 資料與編碼

```
透鏡圖 64×64 → 降採樣 16×16 = 256 像素 → flatten + L2 normalize
→ AmplitudeEmbedding 進 8 qubit (2^8 = 256 basis；每像素 = 一個振幅係數)
```
- 原型可先用 **8×8 → 6 qubit** 加速驗證機制，再放大到 16×16 / 8 qubit。

---

## 3. 遮罩（忠實 QMAE）

- 16×16 切成 **4×4 patch = 16 patch**；遮 **25%**（QMAE 證明 25% 最佳，>50% 變雜訊）。
- **可學習 mask token**：4×4=16 維可訓練參數，在**古典空間**取代被遮 patch，再做 amplitude embedding
  （繞過量子電路中途不能測量/重置的限制）。
- ⚠️ 透鏡圖為「暗背景 + 稀疏弧」，遮罩偏好覆蓋含訊號區，避免老是遮到背景。

---

## 4. 編碼 / 解碼與壓縮

- **Encoder `U(θ)`**：變分電路（原型用 `StronglyEntanglingLayers` 近似 Wang et al. 的雙比特 ansatz；
  忠實版 = 每對 qubit 9 RZ + 6 RY + 3 CNOT = 15 參數/對，8 qubit → 420 參數）。
- 壓縮 `n=8 → k` latent qubit，`t=n−k` 為 **trash**。先試 `k=7,t=1`，再試 `k=4`。
- **Decoder `U†(θ)`**：同參數的伴隨。

### 4.1 trash 重置 + 重建（模擬器中可微分的忠實實作）

```
wires data  = 0..n-1   : AmplitudeEmbedding(masked image)
wires recon = n..2n-1  : 起始 |0...0⟩
U(θ) 作用於 data
SWAP latent 子集 (0..k-1) ↔ (n..n+k-1)   # 把 latent 搬到新暫存；新暫存 trash 位置天然是 |0⟩ → 等效 trash 重置
U†(θ) 作用於 recon
ρ_recon = density_matrix(recon wires)
```

### 4.2 重建 loss（自監督，無標籤）

```
fidelity = ⟨original | ρ_recon | original⟩          # original = 未遮原圖的 amplitude 向量
L_recon  = 1 − fidelity
```
（模擬器可直接由 density matrix 算，免 SWAP-test ancilla；忠實硬體版才用 SWAP test。）
可選加 trash 懲罰 `L_trash = Σ_trash (1 − ⟨Z⟩)/2` 鼓勵壓縮。

---

## 5. 下游三分類（兩種，都配對照）

- **(主) latent → 分類**：量 latent qubit 的 ⟨Z⟩ 當特徵 → 古典小分類頭 → axion/cdm/no_sub。
  測「量子 latent 有沒有判別力」。
- **(次) reconstruct → 分類**：重建圖 → 古典 CNN 分類（QMAE 原文做法）。

---

## 6. 量子位元預算

```
原型(16×16): data 8 + recon 8 = 16 qubit → 2^16 statevector（batch 可模擬）
忠實 SWAP-test 版另加 original 參考 8 + ancilla 1 ≈ 18+（與原文一致）
```
H100 上 `default.qubit` + backprop 可跑（batch 128 約數百 MB）。

---

## 7. 對照組（延續本專案的歸因方法論）

| 組 | 內容 | 回答 |
|---|---|---|
| **DeepLense-QMAE** | 上述量子版 | 量子表現 |
| **Classical-AE sham** | 同壓縮率古典 autoencoder | 量子 vs 同容量古典 |
| **classical MAE (16×16)** | 現有 SimMIM MAE | 對照古典線 |
| **few-shot 掃描** | 每類 N=50/100/500 | 量子最可能顯效的 regime |

---

## 8. 預期結果（寫進 proposal 要誠實）

- 絕對分數低（16×16 + 8-qubit 量子 AE），**不會贏 0.968 古典 MAE**。
- 貢獻 = **第一個 QMAE on 強透鏡 dark matter** + 嚴謹 sham 對照 + few-shot 分析
  + 把 [2511.17372] 從 MNIST 擴展到真實科學資料。

---

## 9. 實作檔案

| 檔案 | 用途 |
|---|---|
| `02_Code/mae-lensing/extract_images16.py` | 下採樣透鏡圖到 16×16，存 train/val 子集 + labels |
| `02_Code/mae-lensing/quantum_mae.py` | QMAE 電路（amplitude embed + mask token + U/U† + 重建 fidelity）|
| `02_Code/mae-lensing/train_qmae.py` | 預訓練重建 → 下游三分類；含 `--sham` 對照 |
| `slurm/run.sbatch` | SLURM 提交（account mst114318，dev 測試 / normal 正式）|

---

## 10. 里程碑

1. ✅ 設計（本文件）
2. ⏳ 最小原型：8×8 或 16×16 跑通「遮罩 → 編碼 → trash 重置 → 解碼 → 重建 fidelity」
3. ⏳ latent → 三分類 + sham 對照
4. ⏳ few-shot data-scaling 曲線
5. ⏳ 整理進 proposal Method / Results
