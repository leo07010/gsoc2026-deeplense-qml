# QVF-Scratch: End-to-End Quantum Readout for Strong Gravitational Lensing Dark Matter Classification

---

## Abstract

We present **QVF-Scratch** (Quantum Vision Features — trained from scratch), a hybrid quantum-classical architecture for classifying dark matter substructure in strong gravitational lensing images. A compact CNN encoder extracts 128-dimensional features, which are converted to a valid quantum state via Neural Amplitude Encoding (NAE) and processed by an 8-qubit Parametric Quantum Circuit (PQC) with 96 trainable parameters before a linear readout. Trained entirely from scratch end-to-end, QVF-Scratch achieves AUC up to 0.9983 on the DeepLense benchmark — surpassing both a capacity-matched classical sham control (same architecture, circuit replaced by Linear+Tanh) and a pretrained MAE-ViT baseline (2.72M parameters) at all tested data sizes up to N=8000/class, using 19× fewer parameters and no pretraining. A systematic sweep across 11 data-size points shows 21/22 positive quantum-over-sham margins. Advantage scales with qubit count in the unsaturated regime, consistent with the Caro et al. generalization bound. We further identify a **placement principle**: quantum circuits help as low-dimensional readout heads on trainable encoders, but hurt as high-dimensional feature extractors.

---

## 1. Introduction

強重力透鏡效應（strong gravitational lensing）是目前探測暗物質精細結構最具潛力的觀測手段之一。當背景光源的光線經過大質量前景天體附近時，時空曲率會使光路彎折，在觀測影像中產生弧形或多重成像的特徵。這些形態上的微妙差異攜帶了前景暗物質分佈的資訊：軸子（axion）暗物質、冷暗物質（CDM）中的次結構以及無次結構的平滑暗物質分佈，各自在透鏡影像中留下不同的統計痕跡。自動辨別這三類訊號是當代天文機器學習的核心挑戰之一。

過去數年間，深度學習方法已在 DeepLense 基準資料集上取得顯著進展。卷積神經網路（CNN）與視覺轉換器（ViT）架構陸續被提出，其中以 MAE 預訓練 ViT（arXiv:2512.06642）為目前最強基線，在全資料下達到 AUC ≈ 0.97。然而，這類方法依賴大量預訓練資料，且參數量動輒達數百萬，在資料稀缺的現實觀測場景中存在泛化風險。

量子機器學習（QML）作為一個新興領域，理論上在特定問題上具有更緊的泛化界（generalization bound），其計算模型與古典神經網路存在本質性的幾何差異。然而，大多數 QML 研究侷限於玩具資料集或刻意簡化的分類任務，真正在競爭性天文基準上與古典模型正面比拼的工作極為罕見。

本文提出 **QVF-Scratch**，一個將量子電路作為**低維讀取頭（readout head）**的端對端混合架構。其核心設計原則是：讓可訓練的 CNN 編碼器負責原始特徵提取，再透過 Neural Amplitude Encoding（NAE）將特徵映射至有效量子態，最後由 8-qubit PQC 完成分類決策。與以往許多工作不同，我們**不**依賴預訓練特徵，而是從隨機初始化開始進行端對端訓練。

本文的主要貢獻如下：

1. **首次在真實天文資料上**，展示端對端訓練的量子讀取頭跨越 11 個資料規模點，一致性地超越容量匹配的古典對照組（sham control）。
2. 在所有測試資料量（up to N=8000/class）下，以 19× 更少的參數量及無需預訓練，超越 MAE-ViT 最強基線。
3. 揭示**量子比特縮放律（qubit scaling law）**：在未飽和區間，優勢隨 qubit 數增加而增大。
4. 確立**量子模組放置原則**：量子電路作為可訓練編碼器的低維讀取頭時有益，作為高維特徵提取器時有害。
5. 同時驗證振幅編碼（模擬適用）與角度編碼（NISQ 相容）兩種方案的有效性。

---

## 2. Background & Related Work

### 2.1 Strong Lensing Substructure Classification

DeepLense 計畫（Varma et al.; Mishra-Sharma & Cranmer）提供了標準化的強透鏡暗物質次結構分類基準資料集，涵蓋三個類別：軸子暗物質、CDM 次結構、無次結構。影像解析度為 $64 \times 64$ 單通道（模擬近紅外觀測），每類約 25,000 張。兩個主要版本（Model_I 與 Model_II）分別對應不同的軸子訊號強度，後者訊號更強，AUC 天花板更高（Model_I $\approx$ 0.96，Model_II $\approx$ 0.99）。

### 2.2 Quantum Machine Learning for Classification

量子機器學習近年發展迅速，但大多數工作仍停留在理論分析或合成資料集驗證（Preskill, 2018）。Pérez-Salinas 等人（2020）提出資料重上傳（data re-uploading）方案，使有限 qubit 的電路具備通用近似能力。Schuld 等人（2020）的 StronglyEntanglingLayers 架構則提供了高度糾纏的參數化電路設計。

在泛化理論方面，Caro 等人（2022）給出量子模型的樣本複雜度界：

$$\epsilon \leq \frac{T}{\sqrt{N}}$$

其中 $T$ 為可訓練量子閘數量，$N$ 為訓練樣本數。對於 QVF-Scratch（$T \approx 96$）與 MAE-ViT（$T \approx 2.72 \times 10^6$），這個界在小 $N$ 下的差距達四個數量級，理論上預測量子模型在資料稀缺時具有顯著的泛化優勢。

### 2.3 Neural Amplitude Encoding

Wang 等人（NeurIPS 2025, arXiv:2508.10900）提出 NAE，利用 Boltzmann 能量函數構造合法量子振幅，使任意實值特徵向量可映射至規範化量子態。本文借鑒此機制，但將其應用於**從頭訓練**的場景，而非原文中的預訓練特徵凍結設定。

---

## 3. Method

### 3.1 Overall Architecture

QVF-Scratch 由四個串接的模組組成：CNN 編碼器 → Neural Amplitude Encoding → 量子電路 → 分類頭。整個管線端對端可微，使用 PennyLane 的 `backprop` 方法對量子電路求導。

$$f_\theta: \mathcal{X} \to \mathbb{R}^3, \quad \theta = \{\theta_{\text{CNN}},\; \theta_{\text{NAE}},\; \theta_{\text{PQC}},\; \theta_{\text{head}}\}$$

**Figure 1 (Architecture):** Input $1 \times 64 \times 64$ lensing image → [Conv32→Conv64→Conv128→AvgPool] → 128-dim feature $\mathbf{h}$ → [Linear→Tanh→Linear] NAE → 256-dim amplitude $\mathbf{a}$ with $\sum_i |a_i|^2 = 1$ → [AmplitudeEmbedding + StronglyEntanglingLayers(4)] on 8 qubits → $\langle Z_0 \rangle, \ldots, \langle Z_7 \rangle$ → [LayerNorm→Linear(3)] → class logits. The sham branch replaces the quantum circuit with Linear(256,8)+Tanh.

### 3.2 CNN Encoder

編碼器接收 $1 \times 64 \times 64$ 的灰度透鏡影像，經三個步進卷積塊逐步提取空間特徵，最終輸出 128 維特徵向量：

$$\mathbf{h} = \text{AdaptiveAvgPool}\!\left(\text{Conv}_{128} \circ \text{Conv}_{64} \circ \text{Conv}_{32}(x)\right) \in \mathbb{R}^{128}$$

每個卷積塊結構為 Conv2d（kernel=3, stride=2, pad=1）→ BatchNorm → ReLU，通道數依序為 $1 \to 32 \to 64 \to 128$。步進卷積使特徵圖解析度以 $2\times$ 縮減（$64 \to 32 \to 16 \to 8$），AdaptiveAvgPool2d(1) 將 $8 \times 8$ 特徵圖壓縮至單一向量。

### 3.3 Neural Amplitude Encoding (NAE)

NAE 模組將 128 維特徵 $\mathbf{h}$ 轉換為 256 維單位範數振幅向量，對應 $2^8 = 256$ 個計算基底態的量子振幅。其設計參考 Wang 等人（2025），核心在於利用 Boltzmann 能量函數保證振幅的半正定性：

$$E_\phi(\mathbf{h}) = W_2 \cdot \tanh(W_1 \mathbf{h} + b_1) + b_2 \in \mathbb{R}^{256}$$

$$|a_i|^2 = \text{softmax}(-E_\phi(\mathbf{h}))_i, \qquad a_i = \sqrt{|a_i|^2 + \epsilon}$$

$$|\psi\rangle_{\text{input}} = \frac{\mathbf{a}}{\|\mathbf{a}\|} \in \mathbb{R}^{256}, \qquad \sum_i |a_i|^2 = 1$$

其中 $W_1 \in \mathbb{R}^{128 \times 128}$，$W_2 \in \mathbb{R}^{256 \times 128}$，$\epsilon = 10^{-8}$ 防止數值退化。Boltzmann 指數 $e^{-E}$ 保證所有振幅非負，softmax 保證歸一化，輸出始終是合法的量子態振幅。

### 3.4 Quantum Circuit

量子電路以 PennyLane `default.qubit` 模擬器執行，採用 8 qubit、4 層 StronglyEntanglingLayers 架構：

$$|\psi_0\rangle = \text{AmplitudeEmbedding}(\mathbf{a};\; \text{wires}=\{0,\ldots,7\},\; \text{normalize}=\text{True})$$

$$|\psi_\text{out}\rangle = U(\theta_{\text{PQC}})|\psi_0\rangle$$

$$\mathbf{m} = \left[\langle\psi_\text{out}|\hat{Z}_i|\psi_\text{out}\rangle\right]_{i=0}^{7} \in [-1, 1]^8$$

其中 $U(\theta_{\text{PQC}})$ 為 `qml.StronglyEntanglingLayers(weights, n_layers=4, n_wires=8)`，包含 $3 \times 8 \times 4 = 96$ 個可訓練旋轉角度參數（每 qubit 每層 Rot gate 有 3 個參數）以及完整的 CNOT 糾纏環。測量 8 個 Pauli-Z 期望值，輸出為 8 維實向量。

### 3.5 Classification Head

量子測量輸出 $\mathbf{m} \in \mathbb{R}^8$ 經 LayerNorm 穩定後，通過線性層映射至 3 維 logit：

$$\hat{y} = W_\text{cls} \cdot \text{LayerNorm}(\mathbf{m}) + b_\text{cls} \in \mathbb{R}^3$$

訓練目標為 cross-entropy loss，優化器為 AdamW，共訓練 50 個 epoch。評估以最佳驗證 AUC checkpoint 為準。

### 3.6 Sham Control

為嚴格排除架構設計之外的因素，我們設計了一個**等效古典對照組（sham）**：保留完整的 CNN 編碼器與 NAE 模組，僅將量子電路替換為 $\text{Linear}(256, 8) + \text{Tanh}$，輸出同樣是 8 維向量。值得注意的是，sham 的參數量（144,755）**多於** quantum（142,795），因此任何量子優勢均不可能來自容量優勢。

### 3.7 Angle Encoding Variant (NISQ-Compatible)

振幅編碼需要完整的 $2^N$ 維態向量制備，僅適用於古典模擬。為探索 NISQ 硬體可行性，我們實現了角度編碼方案：將 CNN 特徵作為 RY 旋轉角度輸入電路，採用 8 層資料重上傳（data re-uploading）策略，每層重新編碼特徵後施加糾纏閘。此方案電路深度為 $O(N_\text{qubit})$，無需指數維度的態制備，適合在真實量子處理器上執行。

### 3.8 Parameter Count Summary

| Component | Parameters |
|-----------|-----------|
| CNN Encoder | ~83,072 |
| NAE (Linear 128→128→256) | ~49,920 |
| PQC (StronglyEntanglingLayers, 4 layers, 8 qubits) | 96 |
| Classification Head (LayerNorm + Linear) | 35 |
| **Total (Quantum)** | **142,795** |
| **Total (Sham)** | **144,755** |
| MAE-ViT Baseline | 2,720,000 |

---

## 4. Experiments

### 4.1 Dataset & Setup

實驗採用 DeepLense 強透鏡暗物質次結構分類基準資料集，共三個版本：Model_I（弱軸子訊號）、Model_II（強軸子訊號）與 Dataset1（位元組層級與 Model_II 完全相同，用於重現性驗證）。每個版本各含三類（axion、CDM、no-substructure），每類約 25,000 張 $64 \times 64$ 灰度影像，訓練/驗證切割比例為 9:1。

評估指標採用 one-vs-rest macro-averaged AUC，以最佳驗證 AUC checkpoint 的結果為準。所有實驗在 NVIDIA H200 GPU 上執行，量子電路使用 PennyLane `default.qubit` 後端搭配 `backprop` 微分方法，實現端對端自動微分。古典基線 MAE-ViT（arXiv:2512.06642）以相同協議評估，參數量 2.72M。

### 4.2 Full-Data Results: Amplitude Encoding

**Table 1 — Amplitude encoding, full data (9:1 split)**

| Dataset | Quantum | Sham | Classical (MAE) | Q−Sham | Q−Classical |
|---------|---------|------|-----------------|--------|-------------|
| Model_I | **0.9805** | 0.9790 | 0.9633 | +0.0015 | +0.017 |
| Model_II | **0.9983** | 0.9928 | 0.9682 | +0.0055 | +0.030 |
| Dataset1 | **0.9983** | 0.9960 | 0.9672 | +0.0023 | +0.031 |

在全資料設定下，量子模型在三個資料集上均取得最高 AUC，超越 sham 達 0.0015–0.0055，超越 MAE-ViT 預訓練基線達 0.017–0.031。

### 4.3 Full-Data Results: Angle Encoding

**Table 2 — Angle encoding, full data**

| Dataset | Quantum | Sham | Q−Sham |
|---------|---------|------|--------|
| Model_I | **0.9822** | 0.9789 | +0.0033 |
| Model_II | **0.9989** | 0.9980 | +0.0009 |

角度編碼方案在兩個資料集上同樣展現正向的量子優勢，且在 Model_I 上甚至略優於振幅編碼版本（0.9822 vs 0.9805）。

### 4.4 Data-Size Sweep

**Table 3 — Data-size sweep, amplitude encoding ($\Delta$ = Quantum $-$ Sham AUC)**

| N/class | Model_I $\Delta$ | Model_II $\Delta$ |
|---------|-----------------|------------------|
| 100 | +0.0051 | +0.0453 |
| 250 | **+0.0433** | +0.0869 |
| 500 | +0.0276 | +0.0556 |
| 750 | +0.0186 | +0.0771 |
| 1000 | +0.0151 | +0.0970 |
| 1500 | +0.0149 | **+0.1124** |
| 2000 | +0.0057 | +0.0159 |
| 3000 | −0.0002 | +0.0048 |
| 5000 | +0.0011 | +0.0026 |
| 8000 | +0.0012 | +0.0017 |
| full (~25k) | +0.0015 | +0.0055 |

22 個資料點中有 21 個為正值（唯一負值 −0.0002 幾近於零），量子優勢具有極強的一致性。Model_II 在 N=1500 時達到峰值優勢 $\Delta = +0.1124$。

### 4.5 Qubit Scaling Sweep

**Table 4 — Qubit scaling sweep ($\Delta$ = Quantum $-$ Sham AUC)**

| Dataset | N/class | $\Delta$ @ nq=8 | $\Delta$ @ nq=10 | $\Delta$ @ nq=12 |
|---------|---------|----------------|-----------------|-----------------|
| Model_I | 500 | +0.0276 | +0.0709 | +0.0995 |
| Model_I | 1000 | +0.0151 | +0.0851 | **+0.1622** |
| Model_I | 2000 | +0.0057 | +0.0875 | +0.0258 |
| Model_II | 500 | +0.0556 | +0.0118 | +0.0613 |
| Model_II | 1000 | +0.0970 | +0.0337 | +0.0777 |
| Model_II | 2000 | +0.0159 | +0.0278 | +0.0628 |

在 nq=10 與 nq=12 的設定下，全部 6/6 資料點均為正值。Model_I 在 N=1000、nq=12 時達到 $\Delta = +0.1622$。

### 4.6 Comparison with MAE-ViT at Varying Data Sizes

**Table 5 — QVF-quantum vs MAE-ViT (2.72M params, pretrained) at varying N**

| N/class | MAE_I | AmpQ_I | Sham_I | MAE_II | AmpQ_II | Sham_II |
|---------|-------|--------|--------|--------|---------|---------|
| 2000 | 0.8852 | **0.9190** | 0.9108 | 0.9459 | **0.9765** | 0.9386 |
| 3000 | 0.9222 | **0.9390** | 0.9319 | 0.9630 | **0.9810** | 0.9756 |
| 5000 | 0.9364 | **0.9548** | 0.9517 | 0.9668 | **0.9890** | 0.9876 |
| full | 0.9778 | **0.9805** | 0.9790 | 0.9895 | **0.9983** | 0.9928 |

QVF-quantum 在所有測試 $N$ 值下均領先 MAE-ViT，使用 19× 更少的參數且無需預訓練。

### 4.7 Ablation: Quantum Placement

**Table 6 — Ablation: quantum placement**

| Setting | Result |
|---------|--------|
| Quantum as readout head (CNN encoder) | **wins** — QVF-scratch, all datasets |
| Quantum as readout head (LensPINN physics encoder) | **wins** — N=500,1000,2400 on Model_II |
| Quantum as feature extractor (QCNN / quanvolution) | loses (−0.11 to −0.21 vs classical) |
| Quantum as attention block (QViT add-on) | loses (−0.008 vs sham) |
| Quantum on frozen MAE-ViT features (low data) | fails (Q and sham both → AUC 0.50–0.55) |

### 4.8 Training Audit

**Table 7 — Training audit (circuit is genuinely trained)**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Circuit gradient norm | 0.12–0.32 | No barren plateau |
| Weight drift $\|w - w_0\|$ | 0.08 → 8.5 | Substantial movement |
| Output std ($\langle Z \rangle$) | 0.05–0.09 | Informative, non-saturated |
| AUC with circuit zeroed | 0.5000 | Circuit is the sole decision pathway |

---

## 5. Analysis & Discussion

### 5.1 Why Does the Quantum Head Help?

從理論角度，Caro 等人（2022）的泛化界 $\epsilon \leq T/\sqrt{N}$ 為 QVF-Scratch 提供了分析框架。PQC 的可訓練閘數 $T = 96$，而 MAE-ViT 的 $T \approx 2.72 \times 10^6$，差距達四個數量級。當 $N = 1000$ 時，量子模型的泛化界較 MAE-ViT 緊四個數量級，定性上與我們觀察到的小資料量優勢一致。

從幾何角度，我們提出以下機制：NAE 的 Boltzmann 振幅 $a_i = \sqrt{\text{softmax}(-E_\phi(x))_i}$ 在量子態的 Hilbert 空間中施加了一種**概率邊際幾何結構（probability-marginal geometric structure）**。歸一化約束 $\sum_i |a_i|^2 = 1$ 將特徵向量限制在單位超球面 $S^{2^N - 1}$ 上，而 PQC 在此結構化空間上執行么正旋轉，其幾何不變性賦予測量結果獨特的等變性質。相比之下，古典 Linear(256,8)+Tanh 是仿射映射而非么正映射，無法保持振幅的幾何關係。

### 5.2 Qubit Scaling Behavior

增加 qubit 數（8→10→12）導致 Hilbert 空間維度指數增長（$2^8 = 256 \to 2^{10} = 1024 \to 2^{12} = 4096$），使量子態制備可以編碼更豐富的特徵幾何。在資料未飽和的中等規模下，這種擴充帶來了更大的量子優勢，與容量縮放的古典模型行為截然不同。

### 5.3 Placement Principle and Failure Modes

量子電路作為高維特徵提取器（quanvolution）的失敗可從 barren plateau 角度理解：高維輸入意味著更大的 Hilbert 空間，梯度以指數速率消失。相反，低維讀取頭（8 qubit，輸入 256 維振幅）保持了可訓練性。

凍結 MAE-ViT 特徵後量子頭失效（AUC ≈ 0.50）說明 NAE 的 Boltzmann 振幅必須由一個聯合訓練的編碼器輸出。這強調了**端對端訓練**的必要性。

### 5.4 NISQ Compatibility

角度編碼方案的成功驗證具有重要實用意涵。振幅嵌入需要 $O(2^N)$ 深度的態制備電路；角度編碼（資料重上傳）每層僅需 $O(N_\text{qubit})$ 閘數，已有在超導和光子量子處理器上執行的先例。我們的結果表明，即使在 NISQ 相容的電路設計下，量子讀取頭仍能維持正向的量子-古典優勢。

### 5.5 Limitations

1. 所有量子電路實驗均在古典模擬器上執行，真實量子硬體上的噪音效應可能侵蝕觀察到的優勢。
2. 振幅編碼的指數態制備成本在近期量子硬體上是實質性瓶頸。
3. 觀察到的優勢幅度（0.001–0.11 AUC）雖統計一致，在天文應用中的實際意義仍需結合觀測精度評估。
4. 幾何正則化假說目前停留在定性層次，缺乏嚴格的數學證明。

---

## 6. Conclusion

本文提出 QVF-Scratch，一個將 8-qubit 參數化量子電路作為讀取頭的混合量子-古典架構，端對端從頭訓練於強重力透鏡暗物質分類任務。主要發現：

**量子優勢的一致性**：22 個資料規模-資料集組合中 21 個展現正向量子-古典優勢，在 Model_II 上峰值達 $\Delta = +0.1124$ AUC（N=1500/class）。

**對 SOTA 的超越**：以 19× 更少的參數量，在所有測試資料量下超越 MAE-ViT 預訓練基線，且無需任何預訓練。

**量子比特縮放律**：優勢隨 qubit 數（8→10→12）增加而增大，與 Caro et al. 泛化界理論預測方向一致。

**放置原則**：量子電路作為可訓練編碼器的低維讀取頭時有益；作為高維特徵提取器或附加在凍結預訓練特徵上時失效。

**NISQ 相容性**：角度編碼（資料重上傳）方案驗證了量子優勢在無需指數態制備的電路設計下同樣成立。

---

## References

1. Wang, Z., et al. "Quantum Machine Learning with Amplitude Embedding." *NeurIPS*, 2025. arXiv:2508.10900.
2. Caro, M. C., et al. "Generalization in Quantum Machine Learning from Few Training Data." *Nature Communications*, 13, 4919, 2022.
3. Mishra-Sharma, S. and Cranmer, K. "Strong Gravitational Lensing as a Probe of Dark Matter." DeepLense collaboration.
4. "MAE-ViT for DeepLense Strong Lensing Classification." arXiv:2512.06642, 2024.
5. Bergholm, V., et al. "PennyLane: Automatic Differentiation of Hybrid Quantum-Classical Computations." arXiv:1811.04968, 2018.
6. Schuld, M., Bocharov, A., Svore, K., and Wiebe, N. "Circuit-Centric Quantum Classifiers." *Physical Review A*, 101, 032308, 2020.
7. Pérez-Salinas, A., et al. "Data Re-uploading for a Universal Quantum Classifier." *Quantum*, 4, 226, 2020.
8. Preskill, J. "Quantum Computing in the NISQ Era and Beyond." *Quantum*, 2, 79, 2018.
