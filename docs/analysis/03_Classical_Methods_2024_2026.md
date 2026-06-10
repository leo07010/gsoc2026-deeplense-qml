# 2024–2026 古典方法新進展（暗物質 × 強重力透鏡）

> **目的**：補完先前 `Classical_Methods_DarkMatter_Lensing.md` 中 2024 之前的論文，聚焦近兩年（2024-05 至 2026-05）的新工作
> **覆蓋面**：12 篇新論文，從 substructure detection / classification / inference / simulator / lens finding 五個面向
> **更新日期**：2026-05-27
> **每篇結構**：架構（Architecture）→ 重點（Key Points）→ 缺點（Weaknesses）

---

## 一句話總結（這兩年的趨勢）

> **Transformer / Diffusion 取代 plain CNN**（GraViT, MAE, DiffLense, FlowLensing）；**SBI 從 proof-of-concept 走向實用尺度**（Jarugula 1 萬 lens, Filipp LSST 2500 lens），但同時也暴露 **distributional-shift robustness 問題**（Filipp 2024 / Dhanasingham 2025）；**Domain adaptation 成為顯學**（補 sim-to-real gap）。

---

## 目錄

**A. 監督式 detection / classification**
1. [Tsang, Şengül, Dvorkin 2024 — Substructure Detection in Realistic Systems](#a1-tsang-2024)
2. [GraViT 2026 — ViT/MLP-Mixer Lens Finding](#a2-gravit-2026)
3. [Ancona-Flores 2026 — Dropout for SIE Parameters](#a3-ancona-flores-2026)
4. [MAE Pretraining 2025 — DeepLense SOTA](#a4-mae-2025)

**B. SBI / NPE / NRE 新進展**
5. [Jarugula 2024 — Population-Level Dark Energy via NRE](#b1-jarugula-2024)
6. [Filipp 2024 — Robustness of NRE/NPE to Distributional Shifts](#b2-filipp-2024)
7. [Dhanasingham 2025 — NPE for LOS + Subhalo Populations](#b3-dhanasingham-2025)
8. [Filipp 2026 — LSST NRE Sensitivity Forecast](#b4-filipp-2026)

**C. 生成模型（diffusion / flow matching）**
9. [DiffLense 2024 — Conditional Diffusion Super-Resolution](#c1-difflense-2024)
10. [FlowLensing 2025 — Diffusion Transformer Flow Matching](#c2-flowlensing-2025)

**D. Domain Adaptation**
11. [Schuldt et al. 2024 — DA for HSC Lens Finding](#d1-schuldt-2024)
12. [Ćiprijanović et al. 2024 — MVE + UDA with Uncertainty](#d2-ciprijanovic-2024)

**E. 總結與 GSoC 啟示**
- [新趨勢總表](#e1-新趨勢總表)
- [2024-2026 文獻 gap 分析](#e2-2024-2026-文獻-gap-分析)
- [對你 GSoC proposal 的具體影響](#e3-對你-gsoc-proposal-的具體影響)
- [參考連結](#f-參考連結)

---

# A. 監督式 detection / classification

## A1. Tsang 2024

**論文**：*Substructure Detection in Realistic Strong Lensing Systems with Machine Learning*
**作者**：Arthur Tsang, Atınç Çağan Şengül, Cora Dvorkin
**發表**：arXiv 2401.16624（Jan 2024）

### 架構

- **U-Net image segmentation**（per-pixel binary mask：subhalo present / not）
- **「Realistic」資料設置**：
  - **真實 source**：COSMOS 星系影像（不是 Sérsic toy profile）
  - **真實 lens**：power-law elliptical + multipoles + external shear
  - **realistic noise**：模擬 HST/LSST 雜訊
- 仍是模擬資料，但 source 用實拍的 COSMOS

### 重點

- **第一篇用 COSMOS galaxy 當 source 的 substructure detection 工作**——之前 Ostdiek 2022 都用合成 Sérsic
- **TPR 71% @ FPR 10%** for subhalo $10^9$–$10^{9.5} \, M_\odot$
- High image resolution + high subhalo concentration 表現顯著更好
- 是 Diaz Rivero/Dvorkin/Ostdiek 系列的**第三篇進化**，從 binary detection → SMF → realistic source

### 缺點

| 問題 | 說明 |
|---|---|
| **低濃度 subhalo 完全偵測不到** | $\Lambda$CDM 預期的 typical concentration 太低、U-Net 失效——這是物理上最相關的 regime |
| **仍非 real survey data** | source 真實但 lens 模型仍合成 |
| **單一 mass bin** | 沒像 Ostdiek 2022 那樣推 SMF |
| **與 SBI 路線缺乏比較** | 沒對比同等難度下 SBI 的表現 |

---

## A2. GraViT 2026

**論文**：*GraViT: Transfer Learning with Vision Transformers and MLP-Mixer for Strong Gravitational Lens Discovery*
**作者**：René Parlange, Juan C. Cuevas-Tello, Octavio Valenzuela, Omar de J. Cabrera-Rosas, Tomás Verdugo, Anupreeta More, Anton T. Jaelani
**發表**：*MNRAS* 545, 2 (Jan 2026)；arXiv 2509.00226

### 架構

- **PyTorch pipeline** 系統 benchmark **10 個 transformer / MLP-Mixer 架構**：
  - ViT, DeiT, CaiT, DeiT III, Swin, Twins-SVT, Twins-PCPVT, PiT, CvT, MLP-Mixer
- **預訓練權重**：ImageNet-1k（部分用 ImageNet-21k）
- 三種 fine-tuning 策略：
  1. 只更新 classification head
  2. 解凍一半架構
  3. 整個 unfreeze
- 任務：**lens finding**（二元分類：is this image a strong lens？）

### 重點

- **首篇大規模 ViT family lens finder benchmark**（10 architectures）
- **AUC-ROC > 0.99** on multiple test sets
- **Ensemble 召回率 95.65%** of known lenses
- **MLP-Mixer** 在 aggregated test 上最佳——意外的結果，說明簡單架構在 domain-specific 仍有競爭力
- 用 transfer learning from ImageNet 顯著縮短訓練

### 缺點

| 問題 | 說明 |
|---|---|
| **任務不是 substructure classification** | 是 lens finding（lens vs no-lens），不是 GSoC 三類（CDM/Axion/no-sub） |
| **泛化跨資料集差** | J24 訓的 model 在其他 lens catalog 表現下降——典型 sim-to-real gap |
| **完全沒 dark matter inference** | 只解決 pipeline 第一步（找 lens），下游 substructure 沒做 |
| **計算成本** | 10 個 architecture 全 train 過一遍，工程量大但 novelty 偏中等 |

> **對你 GSoC 的價值**：lens finding 不是你的任務，但 GraViT 的 transformer benchmark 是「現代 ViT in lensing」的最強對照組之一。

---

## A3. Ancona-Flores 2026

**論文**：*Enhancing Gravitational Lens Study with Deep Learning: A Study on Effects of Dropout Regularization*
**作者**：Juan J. Ancona-Flores, A. Hernández-Almada, V. Motta
**發表**：arXiv 2603.06339（Mar 2026）

### 架構

- **修改版 AlexNet**（為什麼用 AlexNet 而不是更新架構未說明，可能 baseline 用途）
- 任務：**lens parameter regression**——預測 Einstein radius、axis ratio、ellipticity（2 components）
- 模擬器：China Space Station Telescope catalog 衍生
- 資料：76,396 張 SIE-profile 模擬影像

### 重點

- **Dropout 是 SIE 參數估計的關鍵**：加 dropout 後 R² ≈ 0.96、SNR ~37 dB
- **相對誤差降 60–76%**——這是非常大的改進
- 90% confidence interval 內最大誤差 ~9%
- 為 lens modeling pipeline 提供具體 dropout 配置建議

### 缺點

| 問題 | 說明 |
|---|---|
| **只用 SIE profile** | 真實 lens 不是純 SIE，需 power-law + multipoles |
| **AlexNet 已過時** | 用 2012 年 architecture 在 2026 年的 paper，沒對比 ResNet/ViT |
| **任務不是 substructure** | regression of lens model parameters，不是 DM classification |
| **純模擬資料** | 沒 sim-to-real |

---

## A4. MAE 2025（已在上一份 md 詳述）

**論文**：*Masked Autoencoder Pretraining on Strong-Lensing Images...*
**arXiv**：2512.06642 (Dec 2025)
**為何在此 list**：因為它是 2024-2026 期間最重要的 DeepLense classification 工作，**AUC 0.968 / acc 88.65%** 仍是當前 SOTA

詳細架構與缺點見上一份 md 的 B2 節。

---

# B. SBI / NPE / NRE 新進展

## B1. Jarugula 2024

**論文**：*Population-level Dark Energy Constraints from Strong Gravitational Lensing using Simulation-Based Inference*
**作者**：Sreevani Jarugula, Brian Nord, Abhijith Gandrakota, Aleksandra Ćiprijanović
**發表**：arXiv 2407.17292（Jul 2024）

### 架構

- **Neural Ratio Estimation (NRE)**——學習 likelihood-to-evidence ratio
- 訓練資料：simulated strong lens images
- 目標：**population-level dark energy equation-of-state $w$**

### 重點

- 把 SBI 從「subhalo mass function」推到 **dark energy** 參數估計
- 在模擬中可把 $w$ 約束到 $1\sigma$
- 目標 application：**4MOST Strong Lensing Spectroscopic Legacy Survey（~10,000 lenses）**
- Population-level 比 single-lens 顯著更精準

### 缺點

| 問題 | 說明 |
|---|---|
| **dark energy 不是 GSoC 題目焦點** | 你的題目是 substructure (CDM/Axion)，不是 cosmology |
| **依賴 4MOST scale** | 假設可拿到 10⁴ lens，當前 dataset 規模還小 |
| **與 substructure inference 相依性** | 推 $w$ 時 substructure 是 nuisance，若 substructure model 錯，bias dark energy |

---

## B2. Filipp 2024

**論文**：*Robustness of Neural Ratio and Posterior Estimators to Distributional Shifts for Population-Level Dark Matter Analysis*
**作者**：Andreas Filipp, Yashar Hezaveh, Laurence Perreault-Levasseur
**發表**：arXiv 2411.05905（Nov 2024）

### 架構

- **不提新 architecture**——對既有 NRE / NPE 做 robustness audit
- 測試方式：訓練分佈外（OOD）的 test data → 量 posterior bias
- 主測試變量：**background source morphology**（最容易 mis-modeled 的 nuisance）

### 重點

- **第一個系統性 NRE/NPE robustness study**
- 關鍵發現：**「in-distribution 表現極好，但 slight OOD shift 就有顯著 bias」**——這對所有 SBI 路徑是警鐘
- 強調 caution 應用於真實天文資料時的危險
- 沒提出 mitigation——是 diagnostic paper

### 缺點

| 問題 | 說明 |
|---|---|
| **無解決方案** | 只報告問題，未提 robustness-improving method（如 domain adaptation, sim-to-real） |
| **僅測 source morphology shift** | 真實 OOD 還有 PSF, noise, lens model 等其他維度 |
| **subhalo population-level focus** | 不是 single-image classification scope |

> **對你 GSoC 的意義**：若你的 quantum method 號稱「robust」，這篇是 reference benchmark——應該做類似 OOD test。

---

## B3. Dhanasingham 2025

**論文**：*Neural posterior estimation of the line-of-sight and subhalo populations in galaxy-scale strong lensing systems*
**作者**：Birendra Dhanasingham, Francis-Yan Cyr-Racine, Daniel Gilman
**發表**：arXiv 2511.17732（Nov 2025）

### 架構

- **NPE**（neural density estimator），預測 lensing parameter posterior
- **核心 novelty**：**同時推 line-of-sight (LOS) halos AND subhalos**——之前都只推其中一個
- 用 power-law parameterization 模擬 multipoles
- 利用 two-point correlation function 的 anisotropic features

### 重點

- 首次把 **LOS halo + subhalo joint inference** 做出來
- 發現重要的 **degeneracy**：LOS halo mass function amplitude vs subhalo mass function normalization
- 揭示 multipole prediction accuracy 是主要 bottleneck

### 缺點

| 問題 | 說明 |
|---|---|
| **作者自承「remains challenging」** | mass function 與 concentration parameter 仍難準確 recover |
| **power-law parameterization 不足** | 對 anisotropic signal 表示力有限 |
| **訓練資料 prior 限制** | 物理 motivated prior 難產生 uniform training set |
| **bus number 一篇 group** | UNM 的工作，缺 cross-validation |

---

## B4. Filipp 2026

**論文**：*LSST Strong Lensing Systems Dark Matter Sensitivity Analysis with Neural Ratio Estimators*
**作者**：Andreas Filipp, Yashar Hezaveh, Laurence Perreault-Levasseur, Daniel Gilman, LSST DESC
**發表**：arXiv 2604.07438（Apr 2026）

### 架構

- **NRE** 套用在 LSST-scale simulated lensing dataset
- 模擬 subhalo + LOS halo 質量低至 $10^7 \, M_\odot$
- LSST realistic observation conditions

### 重點

- **first LSST-specific forecast**：
  - **2500 lenses → 排除 74% prior volume @ 3σ、36% @ 5σ**
  - few-hundred lenses 已可達 Ly-α forest 等級的約束
- 為 LSST 觀測規劃提供 quantitative target

### 缺點

| 問題 | 說明 |
|---|---|
| **「assumes perfect knowledge of data-generating process」** | 作者自承不能直接套到真實資料 |
| **theoretical exercise** | 是 forecast，不是 application |
| **與 Filipp 2024 robustness paper 內部矛盾** | 同團隊 2024 paper 才剛說 NRE 對 OOD 敏感，這篇又依賴 perfect simulation——需 mitigation |

---

# C. 生成模型（diffusion / flow matching）

## C1. DiffLense 2024

**論文**：*DiffLense: A Conditional Diffusion Model for Super-Resolution of Gravitational Lensing Data*
**作者**：Pranath Reddy, Michael W. Toomey, Hanna Parul, Sergei Gleyzer
**發表**：arXiv 2406.08442；*Mach. Learn.: Sci. Tech.* 5, 045049 (2024)

### 架構

- **Conditional diffusion model**（DDPM 風格）
- **Conditioning**：HSC-SSP 低解析度影像，做 denoising + thresholding 預處理
- **Target domain**：HST 高解析度
- U-Net depth/noise schedule 在 paper 內，abstract 未細節

### 重點

- **第一個 lensing 專用 diffusion super-resolution**
- 訓練資料：HSC + HST paired
- 在 SOTA single-image super-resolution baseline 之上
- 跟你 GSoC 題目同團隊（Reddy, Toomey, Gleyzer 出現在你 paper 1 & 2 的 author list）

### 缺點

| 問題 | 說明 |
|---|---|
| **推論慢** | DDPM 需要 multi-step sampling |
| **依賴 paired data** | 需要 HSC-HST 同源 paired image，dataset 構建貴 |
| **任務是 SR 不是 classification** | 與 GSoC 三類分類不直接相關，但可作 quantum classifier 的 input preprocessing |

---

## C2. FlowLensing 2025

**論文**：*FlowLensing: Simulating Gravitational Lensing with Flow Matching*
**作者**：Hamees Sayed, Pranath Reddy, Michael W. Toomey, Sergei Gleyzer
**發表**：arXiv 2510.07878（Oct 2025）

### 架構

- **Diffusion Transformer (DiT)** + **flow matching** training objective
- 替代傳統 ray-tracing simulator（如 lenstronomy, PyAutoLens）
- 支援多 DM model class + continuous parameters

### 重點

- **200× 加速**：0.36 s/image vs 4.8 s/image classical simulator
- 把 lensing simulation 變成 amortized neural generative model
- 同團隊（Sayed/Reddy/Toomey/Gleyzer），延續 DiffLense 路線
- 對 SBI / NPE pipeline 有戰略意義——若 simulator 太慢，inference 也慢；FlowLensing 解決 simulator bottleneck

### 缺點

| 問題 | 說明 |
|---|---|
| **physics fidelity 待驗證** | 200× 加速可能犧牲精準度；對 high-fidelity 物理推論不適用 |
| **訓練成本高** | DiT 仍需大量 ground-truth simulator data 來訓 |
| **不直接做 inference** | 是 simulator，不是 classifier 或 inference model |
| **未在 inference task 上 end-to-end 驗證** | 沒展示 "FlowLensing-generated training data 訓的 classifier" 表現 |

> **對你 GSoC 的意義**：若你想做大規模 quantum benchmark sweep，FlowLensing 可以快速生成訓練資料。

---

# D. Domain Adaptation

## D1. Schuldt et al. 2024

**論文**：*Domain adaptation in application to gravitational lens finding*
**arXiv**：2410.01203（Oct 2024）

### 架構

- 把 simulated dataset 訓的 CNN/ViT 套到真實 HSC 影像
- 用 **adversarial DA**（DANN-style）對齊 simulator 與真實 feature distribution

### 重點

- **第一個專門針對 lens finding 的 sim-to-real DA**
- 處理 HSC 影像實際遇到的 noise、PSF、artifact 差異
- 對 LSST 等下一代 survey 有實用價值

### 缺點

| 問題 | 說明 |
|---|---|
| **lens finding 不是 substructure classification** | 同 GraViT，任務 mismatch |
| **adversarial DA 訓練不穩** | 對 hyperparameter 敏感 |
| **未做 substructure-level DA** | 對 GSoC 題目最直接需求（CDM vs Axion sim-to-real）沒覆蓋 |

---

## D2. Ćiprijanović et al. 2024

**論文**：*Neural Network Prediction of Strong Lensing Systems with Domain Adaptation and Uncertainty Quantification*
**arXiv**：2411.03334（Nov 2024）

### 架構

- **Mixture of Variance Experts (MVE)** + **unsupervised domain adaptation (UDA)**
- 訓練 source：noiseless simulated
- Target domain：含真實巡天 noise 的 simulated
- 任務：lens parameter regression + uncertainty quantification

### 重點

- **第一個整合 DA + UQ 的 lensing 工作**
- 加 UDA 讓 target domain accuracy **提升約 2 倍**
- MVE 提供 calibrated uncertainty（aleatoric + epistemic 分離）

### 缺點

| 問題 | 說明 |
|---|---|
| **noise-only domain shift** | 真實 sim-to-real gap 還有 PSF、background、galaxy population 等 |
| **regression 不是 classification** | 不直接對應 GSoC 三類任務 |
| **MVE 訓練複雜** | 多個 expert head + variance estimation，工程量大 |

---

# E. 總結與 GSoC 啟示

## E1. 新趨勢總表

| 趨勢 | 證據 | 對你 GSoC 的意義 |
|---|---|---|
| **Transformer 完全取代 plain CNN** | MAE 2025 (SOTA)、GraViT 2026、FlowLensing 2025 | Quantum baseline 必須跟 ViT 比，不能只跟 ResNet 比 |
| **Diffusion / Flow Matching 進入 lensing** | DiffLense 2024, FlowLensing 2025 | 可用作 data augmentation；或考慮 quantum-classical hybrid diffusion |
| **SBI 從 toy 走向 LSST 尺度** | Jarugula 2024（10⁴ lens）、Filipp 2026（2500 lens）| 你的 classification 任務可走相反路：少資料、單張 image 的 quantum |
| **Robustness / OOD 成為焦點** | Filipp 2024, Dhanasingham 2025 | Quantum proposal 要主動 address robustness |
| **Domain adaptation 是基本配備** | Schuldt 2024, Ćiprijanović 2024 | Quantum 路線也應規劃 sim-to-real |
| **Realistic source（COSMOS）取代 toy Sérsic** | Tsang 2024 | 提案應註明會用 realistic source |
| **DeepLense 仍無量子工作** | ML4SCI repo 2024–2026 全 classical | **這是 GSoC 題目存在的核心動機** |

## E2. 2024-2026 文獻 gap 分析

| Gap | 信心 | 證據 |
|---|---|---|
| **沒有任何 2024-2026 paper 用 QML 做 DeepLense classification** | **HIGH** | 上述所有 2024-2026 工作 100% classical |
| **沒有 transformer + equivariant 整合 in lensing** | **HIGH** | MAE 不 equivariant；equivariant 工作只到 2023 |
| **沒有 SBI 用於 single-image 3-class classification** | **HIGH** | SBI 都做 population-level inference，沒做 GSoC 任務 |
| **沒有 diffusion model 直接做 classification** | **HIGH** | DiffLense/FlowLensing 都做 generation 不做 classification |
| **沒有 cross-modal foundation model for lensing** | **HIGH** | AstroCLIP 在 galaxy 上做了，lensing 還沒 |
| **沒有 quantum × MAE / quantum × diffusion 整合** | **HIGH** | 任何 quantum-lensing combo 都還沒有 |

## E3. 對你 GSoC proposal 的具體影響

### 1. Baseline 必須更新

之前你的對照組是 Alexander 2019 ResNet（AUC 0.998 多類），但 2024-2026 後**新 baseline = MAE 2025**（AUC 0.968 / acc 88.65%）。Quantum 路線拿不到 ≥ acc 80% 在這個任務上會很尷尬。

### 2. Robustness 必須是 proposal 的賣點

**Filipp 2024** 給的訊號：所有 NRE/NPE 方法都 OOD 敏感。如果 quantum 路線**對 OOD 更 robust**（因為 entanglement-structured representation 較不依賴特定 nuisance），這是強力 selling point。

### 3. Simulator/generation 不是你的戰場

DiffLense / FlowLensing 已經由 ML4SCI 同團隊（Reddy, Toomey, Gleyzer）做完，不要重做。

### 4. 把 Tsang 2024 的「realistic source」設置納入評估

用 COSMOS source 而非純 Sérsic profile 是現在最低標準，量子方法應展示**在 realistic source 下仍有可比結果**。

### 5. 不要碰 lens finding

GraViT、Schuldt DA 已把 lens finding 做透。你的任務是 substructure classification，scope 要清楚劃分。

---

## F. 參考連結

### 2024 papers
- [Tsang et al. (arXiv 2401.16624)](https://arxiv.org/abs/2401.16624)
- [DiffLense (arXiv 2406.08442)](https://arxiv.org/abs/2406.08442) — *MLST* 5, 045049 (2024)
- [Jarugula et al. (arXiv 2407.17292)](https://arxiv.org/abs/2407.17292)
- [Schuldt et al. DA (arXiv 2410.01203)](https://arxiv.org/abs/2410.01203)
- [Ćiprijanović MVE+UDA (arXiv 2411.03334)](https://arxiv.org/abs/2411.03334)
- [Filipp Robustness (arXiv 2411.05905)](https://arxiv.org/abs/2411.05905)

### 2025 papers
- [GraViT (arXiv 2509.00226)](https://arxiv.org/abs/2509.00226) — *MNRAS* 545, 2 (2026 publish)
- [FlowLensing (arXiv 2510.07878)](https://arxiv.org/abs/2510.07878)
- [Dhanasingham NPE LOS+Subhalo (arXiv 2511.17732)](https://arxiv.org/abs/2511.17732)
- [MAE Pretraining (arXiv 2512.06642)](https://arxiv.org/abs/2512.06642)

### 2026 papers
- [Ancona-Flores Dropout (arXiv 2603.06339)](https://arxiv.org/abs/2603.06339)
- [Filipp LSST NRE (arXiv 2604.07438)](https://arxiv.org/abs/2604.07438)

### ML4SCI / DeepLense 2024-2026 GSoC
- [GSoC 2024 DeepLense projects](https://ml4sci.org/gsoc/projects/2024/project_DEEPLENSE.html)
- [GSoC 2025 DeepLense projects](https://ml4sci.org/gsoc/projects/2025/project_DEEPLENSE.html)
- [DeepLense GitHub repo](https://github.com/ML4SCI/DeepLense)

---

## 附錄：候選 BibTeX

> ⚠️ 作者名與 arXiv ID 已驗證；正式期刊 volume/page 仍以期刊頁面為準。

```bibtex
@article{tsang2024substructure,
  title={Substructure Detection in Realistic Strong Lensing Systems
         with Machine Learning},
  author={Tsang, Arthur and {\c{S}}eng{\"u}l, At{\i}n{\c{c}} {\c{C}}a{\u{g}}an
          and Dvorkin, Cora},
  year={2024},
  archivePrefix={arXiv},
  eprint={2401.16624}
}

@article{reddy2024difflense,
  title={{DiffLense}: A Conditional Diffusion Model for Super-Resolution
         of Gravitational Lensing Data},
  author={Reddy, Pranath and Toomey, Michael W. and Parul, Hanna and
          Gleyzer, Sergei},
  journal={Machine Learning: Science and Technology},
  volume={5},
  number={4},
  pages={045049},
  year={2024},
  archivePrefix={arXiv},
  eprint={2406.08442}
}

@article{jarugula2024population,
  title={Population-level Dark Energy Constraints from Strong Gravitational
         Lensing using Simulation-Based Inference},
  author={Jarugula, Sreevani and Nord, Brian and Gandrakota, Abhijith and
          {\'C}iprijanovi{\'c}, Aleksandra},
  year={2024},
  archivePrefix={arXiv},
  eprint={2407.17292}
}

@article{filipp2024robustness,
  title={Robustness of Neural Ratio and Posterior Estimators to
         Distributional Shifts for Population-Level Dark Matter Analysis
         in Strong Gravitational Lensing},
  author={Filipp, Andreas and Hezaveh, Yashar and Perreault-Levasseur,
          Laurence},
  year={2024},
  archivePrefix={arXiv},
  eprint={2411.05905}
}

@article{sayed2025flowlensing,
  title={{FlowLensing}: Simulating Gravitational Lensing with Flow Matching},
  author={Sayed, Hamees and Reddy, Pranath and Toomey, Michael W. and
          Gleyzer, Sergei},
  year={2025},
  archivePrefix={arXiv},
  eprint={2510.07878}
}

@article{dhanasingham2025neural,
  title={Neural posterior estimation of the line-of-sight and subhalo
         populations in galaxy-scale strong lensing systems},
  author={Dhanasingham, Birendra and Cyr-Racine, Francis-Yan and Gilman,
          Daniel},
  year={2025},
  archivePrefix={arXiv},
  eprint={2511.17732}
}

@article{parlange2026gravit,
  title={{GraViT}: Transfer Learning with Vision Transformers and
         {MLP}-Mixer for Strong Gravitational Lens Discovery},
  author={Parlange, Ren{\'e} and Cuevas-Tello, Juan C. and Valenzuela,
          Octavio and others},
  journal={Monthly Notices of the Royal Astronomical Society},
  volume={545},
  number={2},
  year={2026},
  archivePrefix={arXiv},
  eprint={2509.00226}
}

@article{filipp2026lsst,
  title={{LSST} Strong Lensing Systems Dark Matter Sensitivity Analysis
         with Neural Ratio Estimators},
  author={Filipp, Andreas and Hezaveh, Yashar and Perreault-Levasseur,
          Laurence and Gilman, Daniel},
  year={2026},
  archivePrefix={arXiv},
  eprint={2604.07438}
}
```

---

> **如何在 VSCode 預覽**：開啟此檔案 → `Ctrl+Shift+V` 全螢幕預覽
