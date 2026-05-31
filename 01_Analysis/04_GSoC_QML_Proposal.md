# GSoC 2026 Proposal
## Hybrid Quantum-Classical Representation Learning for Dark Matter Substructure Classification

> **組織**：ML4SCI — DeepLense
> **作者**：[你的名字]
> **聯絡**：leo07010@gmail.com
> **提案日期**：2026-05-27
> **狀態**：草稿 v1

---

## 摘要 (Abstract)

DeepLense 的 classical pipeline 已涵蓋 CNN、ViT、equivariant、SSL、MAE、diffusion、SBI 等所有現代方法，**2024–2026 的 12 篇新工作中 0 篇是 quantum**。本提案在 ML4SCI/DeepLense 體系中建立**第一個系統性的 hybrid quantum-classical 分類 benchmark**，提出三條互補的研究方向：

1. **方向 1（Foundation）**：Quanvolution + 量子核（QSVM）作為快速 baseline
2. **方向 2（Novelty）**：**Equivariant Quantum Convolutional Network (EQCNN)**——把強透鏡的 SO(2) 對稱性寫進 quantum ansatz
3. **方向 3（SOTA-targeting）**：**Quantum-enhanced MAE (Q-MAE)**——把量子分類頭接在 2025 SOTA 的 MAE 預訓練 ViT encoder 後

外加 2 個 stretch goals：Quantum Autoencoder for anomaly detection（延續 Alexander 2021），與 OOD-robust quantum classifier（呼應 Filipp 2024 警鐘）。

評估目標：對標當前 SOTA（**MAE 2025: AUC 0.968 / acc 88.65%**），在固定 parameter budget 下進行公平 head-to-head。Deliverables 不押在「quantum 一定贏」，而是 *characterize where the quantum/entangled ansatz helps and where it doesn't*——即便 negative result 也是有價值的 community contribution。

---

## 目錄

1. [背景與動機](#1-背景與動機)
2. [問題陳述與 Gap 分析](#2-問題陳述與-gap-分析)
3. [方向 1：Quanvolution + QSVM Baseline](#3-方向-1quanvolution--qsvm-baseline)
4. [方向 2：Equivariant Quantum CNN (EQCNN)](#4-方向-2equivariant-quantum-cnn-eqcnn)
5. [方向 3：Quantum-Enhanced MAE (Q-MAE)](#5-方向-3quantum-enhanced-mae-q-mae)
6. [Stretch Goal A：Quantum Autoencoder](#6-stretch-goal-aquantum-autoencoder-for-anomaly-detection)
7. [Stretch Goal B：OOD-Robust Quantum Classifier](#7-stretch-goal-bood-robust-quantum-classifier)
8. [跨方向統一評估協議](#8-跨方向統一評估協議)
9. [時程規劃 (12 週)](#9-時程規劃-12-週)
10. [預期 Deliverables](#10-預期-deliverables)
11. [風險與緩解](#11-風險與緩解)
12. [Why Me / 所需技能](#12-why-me--所需技能)
13. [References](#13-references)

---

## 1. 背景與動機

### 1.1 物理任務

強重力透鏡影像在 lensing arc 邊緣編碼了 sub-percent 等級的子結構簽名，能區分競爭的暗物質模型：

- **No substructure**：純平滑 halo
- **Cold Dark Matter (CDM)**：球狀 subhalo 點狀擾動
- **Axion / Fuzzy DM**：vortex 線狀擾動（superfluid signature）

差別細微到肉眼看不出，但 deep learning 已在 simulated DeepLense benchmark 上達到 macro AUC > 0.99（Alexander 2019 ResNet）、acc 88.65%（MAE 2025）。

### 1.2 為什麼需要 QML

ML4SCI DeepLense GitHub repo 中**所有 sub-projects（含 GSoC 2021-2025）皆為 classical**（CNN、ViT、equivariant、SimCLR/DINO、MAE、diffusion、domain adaptation）。**Quantum 工作只在 sister project QMLHEP（HEP 領域）出現，從未跨界到 lensing**。這代表：

| QML 在本任務的賣點 | 風險聲明 |
|---|---|
| **指數 Hilbert 空間** ($2^n$ for $n$ qubits) | NISQ 量子優勢在影像分類尚無嚴格證明 |
| **Entanglement-structured representation** 可表達 classical kernel 難達的非線性相關 | Barren plateau、encoding bottleneck |
| **NISQ-friendly hybrid** 架構成熟（PennyLane + PyTorch） | Amplitude embedding state-prep 可能 $O(N)$ depth |
| **Lensing 的 SO(2) 對稱性**剛好對應 equivariant quantum ansatz | 離散群近似（$C_8$/$D_4$）不完全等效連續對稱 |

**本提案的 framing**：不押 "quantum advantage"，而是建立第一個 *systematic, fair, parameter-matched benchmark* 在 DeepLense 三類分類任務上。

---

## 2. 問題陳述與 Gap 分析

### 2.1 Gap 矩陣（基於完整 2019–2026 文獻調查）

| Gap | 信心 | 現況 |
|---|---|---|
| ML4SCI DeepLense 完全沒有量子工作 | **HIGH** | repo 結構確認，2024–26 新論文 0 篇 quantum |
| Equivariant QML 從未應用於 lensing | **HIGH** | West 2023 EquivQCNN 只在 MNIST/CIFAR 驗證 |
| 把 MAE pretrained encoder 接 quantum head | **HIGH** | MAE paper 2025-12 太新 |
| Quantum autoencoder 從未用於 lensing | **HIGH** | LHC 有（Ngairangbam 2022），astro 沒有 |
| Quantum 對 OOD robustness 從未被 audit | **HIGH** | Filipp 2024 揭示 classical NRE 嚴重 OOD bias，quantum 待測 |
| Tensor network classifier 從未在 DeepLense baseline | **HIGH** | repo 找不到 |

### 2.2 提案三大方向的設計邏輯

```
Direction 1 (Quanvolution + QSVM)          ── Foundation / Sanity Check ──
        │                                      最便宜的可運行 baseline
        ▼
Direction 2 (Equivariant Quantum CNN)      ── Novelty / Original Contribution ──
        │                                      物理對稱性 + quantum ansatz 首次結合
        ▼
Direction 3 (Q-MAE)                        ── SOTA Targeting ──
                                               接最新 classical SOTA 後段試量子
                                                                    + 
Stretch Goal A (Quantum Autoencoder)       ── 延續 paper 2 unsupervised 路線
Stretch Goal B (OOD-Robust Quantum)        ── 答 Filipp 2024 robustness 警鐘
```

三方向**互補非競爭**：D1 提供 floor、D2 提供 novelty、D3 試 ceiling；任何一方向失敗都有獨立價值，至少一個成功就構成完整 GSoC contribution。

---

## 3. 方向 1：Quanvolution + QSVM Baseline

### 3.1 動機

最便宜、最容易跑通的 hybrid 量子方法。直接複製 Rauf et al. 2026（AstroNet, *PRE*）的架構到 lensing 影像，建立量子 baseline 數字。**這個方向不追求 novelty，追求「proposal 中至少有一個能 reproducibly run 的量子結果」**。

### 3.2 架構設計

```
Input (150×150 lensing image)
  ↓ 預處理：resize → 28×28，min-max normalize 到 [0,1]
Patchify (2×2 stride 2 → 14×14 patches)
  ↓
For each 2×2 patch:
  ├─ Quantum encoding: RY(π·x_j) ⊗ RY(π·x_k) 兩個 qubit
  ├─ RandomLayers (固定，非 trainable)
  ├─ Pauli-Z measurement → expectation values
  └─ Output: 2 channels (one per qubit) → Feature map 14×14×2
  ↓
Classical CNN head:
  Conv2D(32, 3×3, ReLU) → MaxPool(2)
  Conv2D(64, 3×3, ReLU) → MaxPool(2)
  Flatten → Dense(64, ReLU) → Dense(3, Softmax)
```

**並行的 QSVM 變體**：

```
Input
  ↓ Classical CNN encoder (ResNet-18 截取至 layer 3, freeze)
  ↓ Features → PCA / linear projection → 8 維
  ↓ Quantum feature map: U_φ(x) = exp(iπ Σ_i x_i Z_i + π² Σ_{i<j} x_i x_j Z_i Z_j)
  ↓ Kernel: K(x,x') = |⟨0|U_φ(x')† U_φ(x)|0⟩|²
  ↓ Classical SVM with quantum kernel → 3-class output
```

### 3.3 評估指標

| 指標 | Target |
|---|---|
| Macro AUC | ≥ 0.93 (matching Diaz Rivero baseline) |
| Per-class accuracy | ≥ 75% all classes |
| Training time | < 2 hours on simulator (CPU/GPU) |
| Parameter count | Classical head < 100K，Quantum part 4 qubits |

### 3.4 Deliverables

- ✅ PennyLane + PyTorch 完整 pipeline
- ✅ Jupyter notebook 可直接重現
- ✅ 拉到 DeepLense repo 的 `Quantum_Classification/Quanvolution_Baseline/` 資料夾
- ✅ Wiki 文件 + 訓練腳本

### 3.5 風險

| 風險 | 緩解 |
|---|---|
| Quanvolution 表現過低（AUC < 0.85） | 增加 quantum channels（從 2 加到 4–6），加 trainable RandomLayer |
| Simulator 慢 | 用 PennyLane `default.qubit.jax` JIT 編譯；batch 平行 |

---

## 4. 方向 2：Equivariant Quantum CNN (EQCNN)

### 4.1 動機（這是本提案最核心的 novelty）

**強重力透鏡在透鏡軸對稱下保持 SO(2) 不變**。Classical CNN 只有 translation equivariance；E(2)-CNN（NeurIPS ML4PS 2023）已證明在 DeepLense 上加入 $C_8$ rotation equivariance 顯著好過 plain CNN。

**Equivariant Quantum Neural Network** 在 QML 文獻中已有 prototype（West et al. 2023 reflection-equivariant、arXiv 2310.02323 p4m-equivariant），但**從未應用於 strong lensing**。本方向首次將 EQCNN 套用到 DeepLense，同時利用：
- Lensing 的 SO(2) inductive bias
- Quantum entanglement 的高表達力

### 4.2 架構設計

**核心思想**：設計 ansatz 使 $U(g \cdot x) = V(g) U(x) V(g)^\dagger$ 對於 $g \in C_8$。

```
Input image (28×28 resized)
  ↓
Group lifting: 將影像複製 8 份，分別旋轉 0°, 45°, 90°, ..., 315°
  ↓
Embedded into 8-fiber feature map
  ↓
Equivariant Quantum Conv Block:
  ├─ Per-fiber amplitude encoding (8 qubits)
  ├─ Twirled SU(4) gate U_eq = (1/|G|) Σ_{g∈G} V(g) U V(g)†
  ├─ CNOT entanglement between fibers
  └─ Per-fiber measurement → 8 expectation values
  ↓
Group pooling (max or mean over 8 orientations)
  ↓
Classical FC head → 3-class softmax
```

**對照 ablation**：
- (a) Plain QCNN（無 equivariance）
- (b) E(2)-CNN（classical equivariant）
- (c) **EQCNN（quantum equivariant）** ← 我們的方法
- (d) Classical CNN + data augmentation（rotation）

### 4.3 評估指標

| 指標 | Target |
|---|---|
| Macro AUC | ≥ 0.95 (matching E(2)-CNN classical baseline) |
| **Sample efficiency** (低資料 regime) | EQCNN 在 1000 training samples 應顯著優於 plain QCNN |
| Parameter count | 與 classical E(2)-CNN 在相同數量級下對比 |
| **Equivariance verification** | 旋轉 input 的 output 應為對應 representation 變換 |

### 4.4 Deliverables

- ✅ EQCNN PennyLane 實作（含 group twirling 工具）
- ✅ 與 E(2)-CNN（用 `e2cnn` package）公平對比的 benchmark
- ✅ Ablation report：equivariance 在 low-data regime 的價值
- ✅ 至少 2 個 mass-bin / SNR setting 的結果

### 4.5 風險

| 風險 | 緩解 |
|---|---|
| Group twirling 計算成本高 | 用 $C_4$ 開始（4 fibers，10 qubits 內），驗證後升 $C_8$ |
| 真實 lensing 不是完全 SO(2)（lens ellipticity + shear） | 比較 $C_4$ vs $C_8$ vs 無對稱，量化「不完美對稱性」對 EQCNN 的影響 |
| Barren plateau | 採用 layerwise training（Skolik et al. 2021），先訓淺層再加深 |

---

## 5. 方向 3：Quantum-Enhanced MAE (Q-MAE)

### 5.1 動機

當前 DeepLense SOTA 是 **MAE 2025**（arXiv 2512.06642，AUC 0.968 / acc 88.65%）。要在這基礎上挑戰，最務實的做法不是從頭重練 quantum，而是**直接接 MAE pretrained ViT encoder，把分類 head 換成 quantum**。

這個 framing 同時對應到：
- **HQViT 2025**（arXiv 2504.02730）quantum attention 的最新進展
- **HEP-QViT 2024**（arXiv 2402.00776）量子 ViT 在物理事件分類

### 5.2 架構設計

```
Input image (150×150)
  ↓
[Frozen] MAE pretrained ViT-Base encoder (from arXiv 2512.06642 weights)
  ↓
CLS token feature (768-dim)
  ↓ Linear projection → 8 or 16 dim
  ↓
Quantum Classification Head:
  ├─ Amplitude embedding (8 qubits ↔ 256-dim Hilbert space)
  ├─ Variational ansatz: l × (StronglyEntanglingLayer)
  │   每層 3 × n_qubits params
  ├─ Measurement: ⟨Z_i⟩ for i = 0, 1, 2 → 3 logits
  └─ Softmax → 3-class probability
```

**進階變體**：把 MAE decoder 也量子化（quantum reconstruction loss），形成端到端 Q-MAE。

### 5.3 評估指標

| 指標 | Target |
|---|---|
| Macro AUC | ≥ 0.968 (match SOTA) → **stretch: > 0.97** |
| Accuracy | ≥ 88.65% (match SOTA) |
| **Parameter efficiency** | Quantum head 用 < 200 params vs classical MLP 用 ~50K params |
| Fine-tuning time | < 50% of full MAE fine-tuning |

### 5.4 Deliverables

- ✅ 載入 arXiv 2512.06642 pretrained weights 的 pipeline
- ✅ Quantum head 與 classical MLP head（同參數量）head-to-head
- ✅ Quantum head 與 classical MLP head（不同參數量）efficiency frontier
- ✅ 如果 quantum 顯著贏，撰寫 short paper 投稿 NeurIPS 2026 ML4PS workshop

### 5.5 風險

| 風險 | 緩解 |
|---|---|
| MAE weights 未公開 | 自己 retrain MAE（~1 週 GPU 時間），或用相近的 ViT-Small pretrained weights |
| Quantum head 過小無法表達 ViT 後 256-dim feature | 用 data re-uploading 增加表達力；或 stack 多層 |
| 量子優勢被 ViT encoder 主宰 | 報告 parameter-efficiency frontier 而非絕對 accuracy；強調 "lightweight head" use case |

---

## 6. Stretch Goal A：Quantum Autoencoder for Anomaly Detection

### 6.1 動機

延續 **Alexander 2021**（你 paper 2）的 unsupervised 路線——AAE 達到 AUC 0.932 ≈ optimal anomaly detector。**Quantum autoencoder**（Romero 2017、HEP: Ngairangbam 2022 PRD）在 LHC 已驗證**勝過 classical AE 在相同 input space**。本 stretch goal 把它套用到 lensing。

### 6.2 架構（簡述）

```
Input image
  ↓ Classical CNN encoder → latent 8 qubits
  ↓ Quantum Autoencoder:
     ├─ Trash-qubit compression（Romero 2017）
     ├─ Trash-qubit fidelity loss
     └─ 訓練只用 no-substructure 影像
  ↓ Anomaly score = trash-qubit purity
  ↓ Threshold → binary anomaly classifier
```

### 6.3 Target

- AUC ≥ 0.93（與 AAE 平手）
- 計算成本：unsupervised < 200 epochs（vs Alexander 2021 的 500 epochs）

### 6.4 為何是 stretch

- 雖然技術可行，但 unsupervised 路線需要更多 hyperparameter tuning
- 真正的 selling point 是 "quantum AE 有沒有比 classical AAE 在 small-data regime 更穩"——這要時間驗證

---

## 7. Stretch Goal B：OOD-Robust Quantum Classifier

### 7.1 動機

**Filipp 2024**（arXiv 2411.05905）揭示所有 classical NRE/NPE 對 OOD 高度敏感——slight source morphology shift 就有顯著 posterior bias。**這是 community 普遍痛點**。

**假說**：Quantum entanglement-structured representation 對 nuisance shift 較不依賴，因為 representation 不是 over-fit 到單一 manifold。**這是個 testable hypothesis**。

### 7.2 評估設計

```
Train on: Sérsic source profile, idealized PSF, SNR=20
Test on:  COSMOS source (Tsang 2024 style), realistic PSF, SNR∈[10, 30]
                                                                    ↑ OOD
Measure: AUC drop from in-distribution to OOD
         → 比較 Quantum vs Classical model 的 robustness
```

### 7.3 Target

- 對 OOD test set，quantum 的 AUC 下降幅度 < classical 的 50%
- 若 hypothesis 為真：構成 NeurIPS ML4PS 等級 contribution
- 若 hypothesis 為假：仍是有價值的 negative result，避免社群錯誤期待

### 7.4 為何是 stretch

依賴 D1/D2/D3 至少有一個 trained model，且需要重做模擬資料（COSMOS source 整合）。

---

## 8. 跨方向統一評估協議

為了 **fair comparison**，所有方向採用相同協議：

### 8.1 資料集

- **主資料**：DeepLense Model I/II/III public release（HuggingFace / Zenodo）
- 切分：train 80% / val 10% / test 10%
- **OOD set**（for stretch B）：自行用 lenstronomy 重新生成，COSMOS source

### 8.2 統一指標

```python
metrics = {
    'macro_auc': roc_auc_score(y_true, y_pred, average='macro'),
    'per_class_acc': accuracy_per_class(y_true, y_pred),
    'macro_f1': f1_score(y_true, y_pred, average='macro'),
    'ece': expected_calibration_error(y_true, y_pred_probs),  # for uncertainty
    'param_count': count_trainable_params(model),
    'flops': estimate_flops(model),
    'train_time_per_epoch': measure_train_time(),
    'inference_time_per_image': measure_inference_time(),
}
```

### 8.3 Baseline 對照表

每個方向都跟下列 baseline 對比：

| Baseline | 來源 | 預期 AUC |
|---|---|---|
| Random | trivial | 0.5 |
| Logistic regression on raw pixels | 古典 | ~0.7 |
| ResNet-18 (Alexander 2019) | classical | 0.984 (macro) |
| E(2)-CNN (NeurIPS ML4PS 2023) | classical equivariant | ~0.97 |
| MAE (arXiv 2512.06642) | classical SOTA | **0.968** |
| Tensor Network (MPS) | quantum-inspired | TBD（這個方向也是 novelty） |

### 8.4 公平性原則

> **參數量對齊**：當報告「quantum vs classical」差距時，**必須在相同 trainable parameter count** 下比。Quantum circuit 若只有 60 params，要跟 60-param classical MLP 比，不能跟 11M params ResNet 比。

---

## 9. 時程規劃 (12 週)

| 週 | 任務 | Deliverable |
|---|---|---|
| **W1** | Setup、跑通 Alexander 2019 baseline、確認資料與環境 | 重現 ResNet AUC 0.984 |
| **W2** | **方向 1**：Quanvolution + CNN 實作 | 跑通 Rauf-style pipeline |
| **W3** | **方向 1**：QSVM 變體、benchmark | D1 完整結果 + notebook |
| **W4** | **方向 2**：EQCNN 設計（$C_4$ 起步） | 4-qubit EQCNN 跑通 |
| **W5** | **方向 2**：擴展到 $C_8$、與 E(2)-CNN 對比 | D2 head-to-head benchmark |
| **W6** | **Midterm evaluation** + D1/D2 報告 | Midterm report |
| **W7** | **方向 3**：MAE encoder 載入、quantum head 設計 | Q-MAE prototype |
| **W8** | **方向 3**：完整 fine-tune、SOTA 對比 | D3 完整結果 |
| **W9** | **跨方向統一 benchmark**：parameter-matched comparison | Master comparison table |
| **W10** | **Stretch A**（quantum autoencoder） 或 **Stretch B**（OOD test） | 取一深做 |
| **W11** | Documentation、code cleanup、PR 到 DeepLense repo | Pull Request opened |
| **W12** | Final report + 1 篇 NeurIPS ML4PS 2026 workshop submission 草稿 | Final deliverable |

---

## 10. 預期 Deliverables

### 10.1 程式碼

- 完整 PennyLane + PyTorch pipeline，PR 到 `ML4SCI/DeepLense` repo
- 預期目錄結構：
  ```
  DeepLense/
    └── Quantum_Classification/
        ├── D1_Quanvolution_Baseline/
        ├── D2_Equivariant_QCNN/
        ├── D3_Q_MAE/
        ├── Stretch_A_QAE/  (if reached)
        ├── Stretch_B_OOD/   (if reached)
        ├── benchmarks/
        └── notebooks/
  ```

### 10.2 文件

- README.md：每方向獨立 quickstart
- Wiki：架構說明、訓練腳本、結果重現
- Jupyter notebooks：互動式 demo（每方向至少 1 個）

### 10.3 學術產出

- 完整 GSoC final report（部落格文章 + PDF）
- **目標**：1 篇 short paper 投稿 NeurIPS 2026 ML4PS workshop
- **若 D2 或 D3 有顯著結果**：擴充為 full paper 投稿 *Machine Learning: Science and Technology* 或 *Quantum Machine Intelligence*

### 10.4 公開資料

- 訓練好的 model weights（HuggingFace Hub）
- Benchmark dataset 切分（reproducibility）

---

## 11. 風險與緩解

| 風險類別 | 具體風險 | 緩解策略 |
|---|---|---|
| **計算** | 12+ qubit simulation 慢 | 主用 ≤ 10 qubits；用 `default.qubit.jax` JIT 加速；batch parallelize |
| **計算** | Stretch B 需要重做模擬資料 | 限縮至小規模 sanity test，不追求 statistically significant 數字 |
| **學術** | Quantum 全面輸給 classical | proposal 已 framed 為 "characterize" 而非 "win"；negative results 仍有 community value |
| **學術** | Barren plateau 卡死訓練 | layerwise training；初始化技巧；用 QCNN（Cong 2019 證明無 barren plateau） |
| **工程** | MAE 2025 weights 未公開 | 自己 retrain（W7 多預留 2 天）；或用 ImageNet pretrained ViT 替代 |
| **時程** | D1+D2+D3 都做不完 | 優先級：D1 > D2 > D3；至少 D1 + D2 必交，D3 視進度 |
| **同儕競爭** | GSoC 同期多人選 DeepLense | 三方向設計強調 *complementary contribution*；EQCNN 是真正 unique angle |

---

## 12. Why Me / 所需技能

### 12.1 必備（已具備）

- [ ] **PyTorch** 熟練度：能讀寫 ResNet/ViT 標準訓練 loop
- [ ] **CNN 與 ViT** 基礎：理解 convolution、attention、residual connection
- [ ] **線性代數**：unitary matrix、Hilbert space、tensor product
- [ ] **Python 工程**：git、unit test、Jupyter

### 12.2 量子相關（學習計畫）

- [ ] **PennyLane**：W0（GSoC 前）讀完官方 demos 4 篇
- [ ] **Qiskit Machine Learning**：作為備援，了解 API
- [ ] **QCNN / Quanvolution**：手上 Rauf 2026 + Anwar 2025 兩篇 paper 已詳讀
- [ ] **Equivariant QML**：West 2023 + Chang 2023 兩篇 reading list 中

### 12.3 物理背景

- [ ] **強重力透鏡基礎**：Alexander 2019 paper 已讀完前 5 節（含 lens equation）
- [ ] **暗物質模型**：CDM / Axion / Superfluid 區別已理解（Alexander 2019 Section IV）
- [ ] 不需要會推導 metric / geodesic equation——只需理解模擬器輸出影像的意涵

### 12.4 我做過的相關工作

> [此段填入你個人的相關背景，例如：「在 [大學/實驗室] 完成 [專案]，使用 PyTorch 實作 [CNN/Transformer]，達到 [指標]。」「在 [課程] 學過 quantum computing 基礎，完成 [作業]」。]

### 12.5 Pre-application Plan（GSoC 前的 self-evaluation task）

- [ ] **Task 1**：在 DeepLense 公開資料上重現 ResNet-18 baseline，達 AUC ≥ 0.95
- [ ] **Task 2**：用 PennyLane 寫一個最小 2-qubit variational classifier 在 toy MNIST 2-class
- [ ] **Task 3**：把 Task 2 的 quantum classifier 接到 Task 1 的 ResNet 後面，建立 hybrid prototype
- [ ] **GitHub**：把上述 3 個 task 放在公開 repo，附 README，作為 application 的證明

---

## 13. References

### 13.1 古典 baseline（必比對）
1. **Alexander S. et al.** *Deep Learning the Morphology of Dark Matter Substructure.* **ApJ** 893, 15 (2020). arXiv:1909.07346
2. **Alexander S. et al.** *Decoding Dark Matter Substructure without Supervision.* arXiv:2008.12731 (2021)
3. **MAE Strong Lensing** *Masked Autoencoder Pretraining on Strong-Lensing Images for Joint DM Model Classification and Super-Resolution.* arXiv:2512.06642 (Dec 2025) — **current SOTA**
4. **Equivariant NN for DM Morphology** NeurIPS 2023 ML4PS workshop paper #188

### 13.2 量子方法基石
5. **Cong I., Choi S., Lukin M.D.** *Quantum Convolutional Neural Networks.* **Nat. Phys.** 15, 1273 (2019)
6. **Henderson M. et al.** *Quanvolutional Neural Networks.* **Quantum Mach. Intell.** 2:2 (2020)
7. **Pérez-Salinas A. et al.** *Data re-uploading for a universal quantum classifier.* **Quantum** 4, 226 (2020)
8. **Havlíček V. et al.** *Supervised learning with quantum-enhanced feature spaces.* **Nature** 567, 209 (2019)
9. **Rauf A., Amin J., Nabi J.-U.** *Hybrid quantum-classical convolutional neural network for astrophysical object classification.* **Phys. Rev. E** 113, 015302 (2026)
10. **Anwar S. et al.** *Hybrid Quantum-Classical Learning for Multiclass Image Classification.* arXiv:2508.18161 (2025)

### 13.3 Equivariant QML（D2 核心）
11. **West M. et al.** *Reflection Equivariant Quantum Neural Networks for Enhanced Image Classification.* **Mach. Learn.: Sci. Technol.** 4, 035027 (2023)
12. **Chang S.Y. et al.** *Approximately Equivariant Quantum Neural Network for p4m Group Symmetries in Images.* arXiv:2310.02323 (2023)

### 13.4 Quantum 進階變體
13. **Romero J., Olson J.P., Aspuru-Guzik A.** *Quantum autoencoders for efficient compression of quantum data.* **Quantum Sci. Technol.** 2, 045001 (2017)
14. **Ngairangbam V.S., Spannowsky M., Takeuchi M.** *Anomaly detection in high-energy physics using a quantum autoencoder.* **Phys. Rev. D** 105, 095004 (2022)
15. **HQViT** *Hybrid Quantum Vision Transformer.* arXiv:2504.02730 (2025)
16. **Hybrid Quantum ViT for HEP.* arXiv:2402.00776 (2024)

### 13.5 OOD / Robustness（Stretch B 動機）
17. **Filipp A., Hezaveh Y., Perreault-Levasseur L.** *Robustness of NRE/NPE to Distributional Shifts.* arXiv:2411.05905 (Nov 2024)
18. **Tsang A., Şengül A.Ç., Dvorkin C.** *Substructure Detection in Realistic Strong Lensing Systems with ML.* arXiv:2401.16624 (Jan 2024)

### 13.6 工具
- PennyLane: <https://pennylane.ai>
- Qiskit ML: <https://qiskit.org/ecosystem/machine-learning>
- e2cnn (classical equivariant baseline): <https://github.com/QUVA-Lab/e2cnn>
- DeepLense repo: <https://github.com/ML4SCI/DeepLense>
- ML4SCI QMLHEP（量子工作前例）: <https://ml4sci.org/gsoc/projects/2025/project_QMLHEP.html>

---

## 附錄 A：三方向技術對照表

| 維度 | D1: Quanvolution+QSVM | D2: EQCNN | D3: Q-MAE |
|---|---|---|---|
| **Novelty** | 低（複製 Rauf 2026） | **高**（首次 lensing） | 中（首次 quantum + MAE） |
| **實作難度** | 低 | 中-高 | 中 |
| **計算成本** | 低（4 qubits） | 中（8-10 qubits） | 中（8-16 qubits） |
| **預期 AUC** | 0.85–0.93 | 0.93–0.97 | 0.95–0.97+ |
| **是否要 retrain MAE** | 否 | 否 | 是（或用 pretrained） |
| **與 paper 1 對應** | 直接（CNN baseline） | 對照（equivariant CNN） | 對照（ViT SOTA） |
| **與 paper 2 對應** | 弱 | 弱 | 中（共用 encoder） |
| **與 paper 3 對應** | **強**（直接複製） | 弱 | 弱 |
| **與 paper 4 對應** | 中（SU(4) ansatz） | **強**（advanced ansatz） | 中（discarded-qubit recycling） |

---

## 附錄 B：手上 PDF 與本提案對應

```
C:\Users\USER\Downloads\GSoC\
├── Deep Learning the Morphology of Dark Matter Substructure.pdf
│     → 提供 D1/D2/D3 的 baseline AUC 數字（0.984 macro）
├── Decoding Dark Matter Substructure without Supervision.pdf
│     → 提供 Stretch A 的對照組 AUC（AAE 0.932）
├── Hybrid quantum-classical convolutional neural network for
│   astrophysical object classification.pdf
│     → D1 的直接架構模板（Rauf 2026 quanvolution）
└── Hybrid Quantum-Classical Learning for Multiclass Image
    Classification.pdf
      → D2/D3 的 SU(4) ansatz 與 discarded-qubit recycling 技巧
```

---

> **預覽方式**
> - VSCode：`Ctrl+Shift+V`
> - 編譯 HTML：`pandoc GSoC_QML_Proposal.md -f gfm -t html5 -s -c github-markdown.css -o GSoC_QML_Proposal.html`

> **提交前 checklist**
> - [ ] 補上「Why Me」段個人經歷
> - [ ] 完成 pre-application 3 個 self-evaluation tasks
> - [ ] 把 GitHub 公開 repo link 加進「12.4 我做過的相關工作」
> - [ ] Mentor 聯絡確認（透過 ML4SCI Slack）
> - [ ] 確認 GSoC 2026 contributor 申請截止日期
