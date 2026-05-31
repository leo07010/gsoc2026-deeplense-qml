# GSoC 題目分析：Hybrid Quantum-Classical Representation Learning for Dark Matter Substructure Classification

> **組織**：ML4SCI (Machine Learning for Science) — DeepLense
> **任務分類**：強重力透鏡影像 → 暗物質模型分類（CDM / Axion-Fuzzy / no-substructure）
> **本文目的**：題目深度解讀 + 古典與量子方法地景 + GSoC proposal 撰寫建議
> **更新日期**：2026-05-27

---

## 目錄

1. [題目實質在做什麼](#1-題目實質在做什麼)
2. [為什麼 ML4SCI 選 QML 而不是更深的 CNN](#2-為什麼-ml4sci-選-qml-而不是更深的-cnn)
3. [手上四篇 paper 的角色定位](#3-手上四篇-paper-的角色定位)
4. [古典方法地景](#4-古典方法地景)
5. [量子方法地景](#5-量子方法地景)
6. [方法 × 任務適配矩陣](#6-方法--任務適配矩陣)
7. [Gap 分析](#7-gap-分析)
8. [Top 5 必讀文獻](#8-top-5-必讀文獻)
9. [Proposal 撰寫建議](#9-proposal-撰寫建議)
10. [所需技能與工具棧](#10-所需技能與工具棧)
11. [風險提醒](#11-風險提醒)
12. [參考連結](#12-參考連結)

---

## 1. 題目實質在做什麼

**輸入**：強重力透鏡（strong gravitational lensing）模擬影像（通常 ~150×150 像素灰階）

**輸出**：三類分類
- **No substructure**：純平滑 halo，無子結構
- **Cold Dark Matter (CDM)**：subhalo 呈球狀的點狀擾動
- **Axion / Fuzzy DM (Superfluid)**：渦旋（vortex）線狀擾動

差別非常細微——肉眼幾乎看不出，只在 lensing arc 邊緣有 sub-percent 等級的扭曲。這是任務「值得試 QML」的關鍵：**特徵很高維、訊號很稀薄、且空間相關性可能跨整張影像**，剛好是 quantum feature map 宣稱優勢的場景。

### 物理示意

```
背景星系 (Source)
     ↓ 光線被透鏡星系彎曲
透鏡星系 + Dark Matter Halo (含或不含 substructure)
     ↓
觀測影像 (扭曲的 arc / Einstein ring)
     ↓ 子結構在 arc 邊緣留下微弱簽名
     ↓
   分類器需從這簽名反推 DM 模型
```

---

## 2. 為什麼 ML4SCI 選 QML 而不是更深的 CNN

從 ML4SCI/DeepLense 的工作累積來看：

- **paper 1**（Alexander 2019）已證明：classical CNN 足以區分 vortex vs WIMP substructure
- **paper 2**（Decoding without Supervision, 2021）展示 unsupervised（autoencoder）也可行
- **DeepLense GitHub repo** 中已涵蓋 ResNet、ViT、equivariant CNN、SSL（SimCLR/DINO）、MAE、diffusion、physics-informed transformer 等所有現代方法

**所以純 supervised CNN 不再是 novel contribution**。QML 的賣點：

| 賣點 | 實質意義 |
|---|---|
| Exponential Hilbert space | n qubits 對應 $2^n$ 維特徵空間 |
| Entanglement | 可表達 classical kernel 難以表達的非線性相關 |
| NISQ 友善的 hybrid | quantum part 只當 feature extractor，反向傳播交給 classical backbone |

> ⚠️ **要實話實說的限制**
> - 目前 NISQ 量子優勢在 image classification 上**沒有嚴格證據**
> - Barren plateaus、noise、encoding bottleneck（amplitude embedding 把 N 像素塞進 $\log_2 N$ qubits，但編碼電路本身可能比後面的 ansatz 還深）
> - Proposal 千萬不要寫「show quantum advantage」，要寫「systematically characterize where hybrid models match/diverge from classical baselines」

---

## 3. 手上四篇 paper 的角色定位

| Paper | 角色 | 你能拿來用什麼 |
|---|---|---|
| **Alexander 2019** *Deep Learning Morphology* | 任務 baseline | CNN 架構、模擬資料生成方式（lenstronomy / PyAutoLens）、CDM vs vortex 物理 |
| **Alexander 2021** *Decoding without Supervision* | 任務延伸 | autoencoder + anomaly detection；可做 QML 版的 quantum autoencoder |
| **Rauf et al. 2026** *AstroNet* (PRE) | **最接近的 QML 範本** | 2×2 patch quanvolution + RandomLayers + classical CNN——架構幾乎可以直接搬到 lensing 圖上 |
| **Anwar et al. 2025** *Hybrid QCNN multiclass* | 進階方法 | SU(4) 卷積 ansatz、amplitude vs angle embedding、回收被 pooling 丟掉的 qubits |

**故事線**：
> 沿用 DeepLense 既有的分類任務 → 把 Rauf 的 quanvolution feature extractor 接到 ResNet/EfficientNet → 加上 Anwar 的 discarded-qubit recycling 與 SU(4) 卷積做 ablation → 與 classical baseline 同 parameter budget 做公平比較。

---

## 4. 古典方法地景

### 4a. 已經在 DeepLense 上做過的（從 ML4SCI GitHub 確認）

| 方法家族 | 代表工作 | 在 DeepLense 表現 | 程式碼位置 |
|---|---|---|---|
| **CNN baseline** (ResNet/EfficientNet) | Alexander et al. 2019 | ~80–90% acc | DeepLense repo |
| **Vision Transformer** | Sachdev / D. Srivastava (GSoC 2021–23) | comparable to CNN | DeepLense repo |
| **Equivariant CNN** (E(2)-CNN, p4m) | Singh (GSoC 2021), NeurIPS ML4PS 2023 | 優於非對稱 CNN | DeepLense repo |
| **Self-supervised (SimCLR/BYOL/DINO/iBOT/SimSiam)** | Deshmukh, Iyer (GSoC 2023–24)；LenSiam (Toomey 2023) | 強過 supervised baseline | DeepLense repo |
| **Masked Autoencoder (MAE)** | arXiv 2512.06642 (Dec 2025) — **目前 SOTA** | **AUC 0.968 / acc 88.65%**（90% mask ratio）vs ViT-from-scratch 0.957 / 82.46% | 論文有附 |
| **Physics-informed Transformer** | Lucas Jose, Anirudh Shankar (GSoC 2023–24) | + GradCAM 可解釋性 | DeepLense repo |
| **Diffusion 模型** | Atal Gupta, Pranath Reddy, Hamees, Difflense | 主要用於 super-resolution & simulation，**未當分類 backbone** | DeepLense repo |
| **Domain Adaptation** | Tidball, Nath (GSoC 2022–23) | sim→real transfer | DeepLense repo |
| **Anomaly detection (Autoencoder/VAE)** | Alexander et al. 2021 | unsupervised baseline | — |

### 4b. 古典側仍未被嘗試（**可填補的 gap**）

| 方法 | 為何適合 | 風險 |
|---|---|---|
| **Simulation-Based Inference (NPE/TMNRE)** Brehmer 2019, Coogan 2024 | 已在 subhalo population inference 上用，但**未在 model classification 上應用**；可給 calibrated posterior 而非單一 label | 需要 differentiable simulator 介面 |
| **Diffusion-as-Feature-Extractor** | DDPM 中間 noise prediction 是強 representation；DeepLense 已有 diffusion model，可 freeze 抽 feature | 推論慢；尚未在科學影像驗證 |
| **Equivariant Vision Transformer** | 結合 ViT 全局 attention + 旋轉等變性 | 程式碼複雜 |
| **AstroCLIP-style cross-modal** | lensing image ↔ lens galaxy spectrum 對比學習 | 需要 paired spectra |
| **Bayesian Deep Ensembles / SWAG** | 對 lensing 分類加 calibrated uncertainty | 純粹工程整合，非 novel method |

---

## 5. 量子方法地景

### 5a. 已在 NISQ 影像/物理分類驗證過

| 方法 | 開創性論文 | 關鍵特性 | 對本任務潛力 |
|---|---|---|---|
| **QCNN** | Cong, Choi, Lukin, *Nat. Phys.* 2019 | 對數深度、無 barren plateau 之證明 | ⭐⭐⭐ 必跑 baseline |
| **Quanvolutional NN** | Henderson et al. *QMI* 2020；Rauf 2026 | 2×2 patch → 2-qubit circuit → multi-channel feature map | ⭐⭐⭐ 最容易接到 lensing 影像 |
| **Hybrid QCNN + SU(4) + recycling** | Anwar et al. arXiv 2508.18161 | SU(4) 通用 2-qubit gate；回收 pool 丟掉的 qubit | ⭐⭐ 進階 ablation |
| **Quantum Kernel / QSVM** | Havlíček et al. *Nature* 2019；Peters et al. *PRX* 2021 | classical SVM + quantum feature map | ⭐⭐⭐ 最便宜的 baseline |
| **Data Re-uploading Classifier** | Pérez-Salinas et al. *Quantum* 2020 | 單一/少 qubit 反覆 encode；NISQ 最 friendly | ⭐⭐ 適合 ultra-low-parameter sweep |
| **Quantum Autoencoder** | Romero et al. *QST* 2017；HEP: Ngairangbam et al. *PRD* 2022 | unsupervised；LHC 已驗證壓 classical AE | ⭐⭐⭐ **延伸 paper 2 路線** |
| **Equivariant Quantum NN（p4m / reflection）** | West et al. arXiv 2310.02323；Chang et al. *MLST* 2023 | 把旋轉/反射 symmetry 寫進 ansatz | ⭐⭐⭐⭐ **最強 novelty 點** |
| **Quantum Vision Transformer** | HQViT (arXiv 2504.02730)；HEP-QViT (arXiv 2402.00776) | quantum attention / quantum MLP | ⭐⭐⭐ 接到 MAE pretrained ViT 後段 |
| **Quantum Contrastive Learning** | Nanda (ML4SCI GSoC 2024, QMLHEP) | 量子 encoder + classical projection head | ⭐⭐⭐ 直接搬到 LenSiam pipeline |
| **Quantum Diffusion Model** | Quark-gluon (arXiv 2412.21082)；medical (2508.09903) | hybrid latent diffusion | ⭐ 偏 generative，不直接分類 |
| **Quantum Reservoir / QELM** | Mujal et al. *PRR* 2023；QuEra 108-qubit (2024) | 只訓 readout，硬體最便宜 | ⭐ 適合 quick-feasibility check |

### 5b. 量子啟發（Quantum-Inspired，純古典硬體可跑）

| 方法 | 為何納入 |
|---|---|
| **MPS / TTN classifier** (Stoudenmire 2016；MNIST ~98%) | 「假量子」baseline，證明 entanglement-style representation 本身能不能加分；訓練快 |
| **MERA 2D** for image | 多尺度，類似 wavelet；可比 quantum CNN 結構但跑 classical |
| **Tensor Train Layer** in CNN | 取代 dense layer，壓參數 |

> 💡 **為什麼 quantum-inspired 重要**
> 它能告訴你：**所謂 "quantum advantage" 是 entanglement-structured ansatz 帶來的，還是 quantum hardware 帶來的**。
> - 如果 MPS 已經比 ResNet 好 → 「quantum」的功勞不在硬體
> - 如果 MPS 跟 ResNet 一樣 → QCNN 多出的部份就值得 highlight

---

## 6. 方法 × 任務適配矩陣

| 方法 | 表達力 | 是否利用 lensing 旋轉對稱 | 與 DeepLense pipeline 整合難度 | NISQ 可行 | 對 GSoC proposal 差異化價值 |
|---|---|---|---|---|---|
| Quanvolution + CNN (Rauf-style) | 中 | ✗ | 低 | ✓ | 必做 baseline |
| **Equivariant QCNN (p4m)** | 中-高 | ✓✓ | 中 | ✓ | **最強原創點** |
| Hybrid QCNN + SU(4) + recycling | 高 | ✗ | 中 | ✓ | 進階 ablation |
| Quantum kernel / QSVM on CNN features | 中 | ✗ | 極低 | ✓ | 便宜的 sanity check |
| Quantum Autoencoder (unsupervised) | 中 | ✗ | 中 | ✓ | **接續 paper 2 路線** |
| Data re-uploading classifier | 低 | ✗ | 低 | ✓✓ | low-parameter regime 對照 |
| Quantum Contrastive (Nanda 風格) | 中-高 | 可加入 | 中 | ✓ | 跟 DeepLense SSL 工作呼應 |
| HQViT / Quantum ViT | 高 | ✗ | 高 | △ (需要更多 qubit) | trendy，stretch goal |
| MPS / MERA (quantum-inspired) | 中 | ✗ | 中 | N/A（純古典） | **量化「真量子是否帶來價值」的 control** |
| Quantum reservoir / QELM | 低-中 | ✗ | 低 | ✓ | 硬體 demo |

---

## 7. Gap 分析

| Gap | 信心 | 證據 |
|---|---|---|
| ML4SCI DeepLense 完全沒有量子工作 | **HIGH** | GitHub repo 結構確認，QML 都在 QMLHEP |
| Equivariant QML 從未應用於 lensing | **HIGH** | EquivQCNN paper 只在 MNIST/CIFAR 驗證 |
| Quantum unsupervised / anomaly detection 從未用於 lensing | **HIGH** | LHC 有，astro 沒有 |
| Tensor network classifier 從未在 DeepLense baseline | **HIGH** | 在 GitHub repo 找不到 |
| Quantum-classical fair comparison（同 parameter budget）在 lensing 沒做過 | **HIGH** | classical 多用 ResNet (~11M params)，QCNN 通常 < 1K params，從未公平對比 |
| 把 MAE pretrained encoder 接 quantum classifier head | **MEDIUM** | MAE paper 太新（2025/12），量子接續工作還沒出現 |

---

## 8. Top 5 必讀文獻

1. **Cong, Choi, Lukin** — *Quantum Convolutional Neural Networks*, *Nature Physics* 15, 1273 (2019)
   - QCNN 經典，含 barren-plateau-free 證明

2. **West, Tsang et al.** — *Reflection Equivariant Quantum Neural Networks for Enhanced Image Classification*, *Mach. Learn.: Sci. Technol.* 4 035027 (2023)
   - Equivariant QML 的入門範本

3. **Pérez-Salinas et al.** — *Data re-uploading for a universal quantum classifier*, *Quantum* 4, 226 (2020)
   - NISQ 最 efficient classifier

4. **Ngairangbam, Spannowsky, Takeuchi** — *Anomaly detection in high-energy physics using a quantum autoencoder*, *Phys. Rev. D* 105, 095004 (2022)
   - HEP 對 quantum AE 的最佳示範

5. **arXiv 2512.06642** — *Masked Autoencoder Pretraining on Strong-Lensing Images for Joint Dark-Matter Model Classification and Super-Resolution* (Dec 2025)
   - **必比對的 SOTA**

備選：
- **Henderson et al.** *Quanvolutional Neural Networks*, *Quantum Machine Intelligence* 2:2 (2020)
- **Stoudenmire & Schwab** *Supervised Learning with Quantum-Inspired Tensor Networks*, NeurIPS 2016

---

## 9. Proposal 撰寫建議

### 推薦故事線（基於 gap 分析）

> 「DeepLense 的 classical pipeline 已經涵蓋 CNN、ViT、equivariant、SSL、diffusion、MAE 等所有現代方法。**唯一沒被探索的維度是量子計算**。本提案建立第一個 fair benchmark：在固定 parameter budget 下，比較：
> 1. classical ResNet/ViT
> 2. tensor-network classifier
> 3. quanvolution + CNN
> 4. **equivariant QCNN（針對 lensing 的 SO(2) symmetry）**
> 5. quantum-augmented MAE
>
> Deliverables 不押在「quantum 一定贏」，而是 *characterize where the quantum/entangled ansatz helps and where it doesn't*。」

這個 framing 同時對應到：
- ✅ paper 1, 2 的 classification + unsupervised 兩條路線
- ✅ paper 3 的 quanvolution baseline
- ✅ paper 4 的 hybrid QCNN 進階做法
- ✅ 加上 equivariant QML 這個「DeepLense 未做、QML 文獻有但未跨界」的真正 novel contribution

### Proposal 應包含的核心要素

**1. Baseline 必須先打**
- 復現 paper 1 的 CNN 數字（ResNet-18 / EfficientNet-B0）
- 在同樣 dataset（DeepLense 公開的 Model I/II/III）上記錄 AUC、per-class accuracy

**2. Hybrid 架構至少 2–3 個變體**

| 變體 | 架構 |
|---|---|
| (a) Quanvolution 前端 | Rauf 風格，2 qubits/patch → classical CNN 後段 |
| (b) Quantum embedding + classical head | CNN 抽特徵 → 降到 8–16 維 → angle/amplitude embedding → variational circuit → measurement → softmax |
| (c) 完整 QCNN | Anwar 風格 SU(4) 卷積 + pooling + 回收 discarded qubits |
| (d) **Equivariant QCNN** | p4m / reflection-equivariant ansatz，對應 lensing 對稱性 |

**3. 公平比較**
- **重點：參數量對齊**
- Quantum circuit 若只有 60 參數，要跟 60 參數的 classical MLP 比，不要跟 11M 參數的 ResNet 比
- 用 **PennyLane** 或 Qiskit + PyTorch interface（`qml.qnn.TorchLayer`）
- Noise model：先 noiseless simulator，再加 depolarizing noise 看 robustness

**4. Domain-specific 觀察點**
- Lensing arc 的旋轉對稱性 → equivariant quantum circuit
- Vortex substructure 的訊號可能集中在 ring 上 → 不該對整張圖均勻 patchify，而是 sample on ring（這個 insight 是 GSoC 上能加分的點）

---

## 10. 所需技能與工具棧

### 必備
- **PyTorch**
- CNN 訓練經驗
- Linear algebra（會看 Hilbert space / unitary）

### 量子
- **PennyLane**（最常用於 hybrid，autodiff 與 PyTorch 整合最好）
- 或 Qiskit Machine Learning

### 物理
- 不需要會推導 lens equation
- 但要看得懂 lensing 影像在物理上代表什麼（paper 1 已涵蓋）

### 資料
- DeepLense 在 HuggingFace / Zenodo 上有公開 dataset
- **強烈建議 application 前先下載跑通 paper 1 的 baseline**——ML4SCI evaluation task 通常就是這個

---

## 11. 風險提醒

> 1. **「Quantum 比 classical 好」的結果很難拿到**——做出來通常是 comparable 或略差。Proposal 要避免把 deliverable 寫成「show quantum advantage」，改寫成「systematically characterize where hybrid models match/diverge from classical baselines」，這樣即便結果是 negative 也是有價值的貢獻。
>
> 2. **Simulator 慢**：12+ qubits 的 statevector simulation 在 CPU 上單張影像可能要秒級，batch training 會痛苦。要規劃好 qubit 數上限（多半 ≤ 10）。
>
> 3. **ML4SCI 是熱門 org**：DeepLense 子題目競爭激烈。差異化建議是把**物理 inductive bias（旋轉對稱、ring sampling）**寫進 quantum circuit 設計，而不只是套現成 architecture。
>
> 4. **Amplitude embedding 的隱藏成本**：把 N 像素塞進 $\log_2 N$ qubits 是 NISQ「賣點」，但實際上 amplitude state preparation circuit 的 depth 可能是 $O(N)$，反而抵銷量子優勢。Proposal 要展示對這個 trade-off 的理解。

---

## 12. 參考連結

### ML4SCI / DeepLense 官方資源
- [ML4SCI DeepLense GitHub repo](https://github.com/ML4SCI/DeepLense)
- [GSoC 2025 DeepLense projects](https://ml4sci.org/gsoc/projects/2025/project_DEEPLENSE.html)
- [GSoC 2025 QMLHEP — Equivariant QNN proposal](https://ml4sci.org/gsoc/2025/proposal_QMLHEP4.html)
- [Sanya Nanda — Quantum Contrastive Learning GSoC 2024](https://sanyananda.github.io/ML4Sci_QuantumContrastiveLearning/)

### 古典方法關鍵論文
- [Alexander 2019 — Deep Learning Morphology of DM Substructure (arXiv 1909.07346)](https://arxiv.org/pdf/1909.07346)
- [Alexander 2021 — Decoding without Supervision (arXiv 2008.12731)](https://arxiv.org/abs/2008.12731)
- [MAE on strong lensing — current SOTA (arXiv 2512.06642)](https://arxiv.org/abs/2512.06642)
- [LenSiam: SSL on Strong Lensing](https://openreview.net/pdf?id=xww53DuKJO)
- [Equivariant Neural Nets for DM Morphology (NeurIPS ML4PS 2023)](https://neurips.cc/virtual/2023/76224)
- [AstroCLIP — cross-modal foundation model](https://academic.oup.com/mnras/article/531/4/4990/7697182)

### 量子方法關鍵論文
- [Cong et al. QCNN (Nature Physics 2019)](https://www.nature.com/articles/s41567-019-0648-8)
- [Henderson Quanvolutional NN (QMI 2020)](https://link.springer.com/article/10.1007/s42484-020-00012-y)
- [Reflection Equivariant QNN (arXiv 2212.00264)](https://arxiv.org/html/2212.00264)
- [Approximately Equivariant QNN for p4m (arXiv 2310.02323)](https://arxiv.org/pdf/2310.02323)
- [HQViT: Hybrid Quantum Vision Transformer (arXiv 2504.02730)](https://arxiv.org/abs/2504.02730)
- [Hybrid Quantum ViT for HEP (arXiv 2402.00776)](https://arxiv.org/abs/2402.00776)
- [Data Re-uploading Classifier (Quantum 2020)](https://quantum-journal.org/papers/q-2020-02-06-226/)
- [Quantum Autoencoder for HEP Anomaly Detection (PRD 2022)](https://link.aps.org/doi/10.1103/PhysRevD.105.095004)
- [Tensor Network for ML (MPS/MERA review)](https://arxiv.org/pdf/2101.03154)
- [Lorentz-Equivariant QGNN (arXiv 2411.01641)](https://arxiv.org/html/2411.01641v1)

### 實作限制與技術細節
- [Limitations of Amplitude Encoding (arXiv 2503.01545)](https://arxiv.org/html/2503.01545v1)
- [Barren Plateau mitigation survey (arXiv 2406.14285)](https://arxiv.org/pdf/2406.14285)

---

## 附錄：手上四篇 PDF 對應

```
C:\Users\USER\Downloads\GSoC\
├── Deep Learning the Morphology of Dark Matter Substructure.pdf
│     → Alexander 2019, classification baseline
├── Decoding Dark Matter Substructure without Supervision.pdf
│     → Alexander 2021, autoencoder + anomaly detection
├── Hybrid quantum-classical convolutional neural network for astrophysical
│   object classification.pdf
│     → Rauf et al. 2026 (PRE), AstroNet — quanvolution + CNN
└── Hybrid Quantum-Classical Learning for Multiclass Image Classification.pdf
      → Anwar et al. 2025, SU(4) QCNN + discarded qubit recycling
```

---

> **如何在 VSCode 預覽這份文件**
>
> 1. 在 VSCode 開啟此檔案
> 2. 按 `Ctrl+Shift+V` → 右側預覽
> 3. 或按 `Ctrl+K V` → 旁邊開預覽視窗（可邊看邊改）
