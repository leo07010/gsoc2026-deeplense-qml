# Tesi et al. 2024 — Quantum Attention for Vision Transformers in HEP

> **重要性**：⭐⭐⭐⭐⭐ — **Sergei Gleyzer 是共同作者**（ML4SCI/DeepLense 主理人；你 paper 1, 2 的 author）
> **分析日期**：2026-05-28
> **PDF 位置**：[`../00_Papers/Quantum_ML/04_Tesi_2024_Quantum_Attention_ViT_HEP.pdf`](../00_Papers/Quantum_ML/04_Tesi_2024_Quantum_Attention_ViT_HEP.pdf)

---

## 0. 為什麼這篇對你 GSoC 是關鍵

| 連結 | 證據 |
|---|---|
| **同團隊** | Sergei Gleyzer 是共同作者，他帶 ML4SCI 整個 QMLHEP + DeepLense 體系 |
| **方法直接可移植** | QONN attention 可套用到 MAE 的 ViT encoder |
| **誠實 framing 的範本** | 結論「quantum 跟 classical 平手」——proposal 寫法可參考 |
| **CMS Open Data 對應 DeepLense** | 都是 multi-channel 科學影像（jet 3 channels vs lensing 1 channel）|
| **paper 3 Pasquali 2024 的延伸** | 同方向但用 QONN 而非 naive QViT |

---

## 1. 論文身分

| 項目 | 內容 |
|---|---|
| **完整標題** | *Quantum Attention for Vision Transformers in High Energy Physics* |
| **arXiv** | [2411.13520](https://arxiv.org/abs/2411.13520) (submitted Nov 20, 2024) |
| **作者** (7 人) | Alessandro Tesi, Gopal Ramesh Dahale, **Sergei Gleyzer**, Kyoungchul Kong, Tom Magorsch, Konstantin T. Matchev, Katia Matcheva |
| **發表** | arXiv preprint（截至目前無正式 venue）|
| **任務** | Quark vs gluon jet classification（HEP 二元分類）|
| **資料** | CMS Open Data multi-detector jet images |

---

## 2. 核心方法：Quantum Orthogonal Neural Network (QONN)

### 2.1 為什麼是「Orthogonal」

| 性質 | 物理意義 |
|---|---|
| **正交矩陣 = norm-preserving** | 跟 amplitude encoding（要求 $\\|x\\|=1$）天然相容 |
| **正交矩陣自由度 = $n(n-1)/2$** | 用 $n(n-1)/2$ 個 RBS gate 剛好對應（無冗餘）|
| **梯度穩定** | 不會 vanishing/exploding（unitary 子集）|
| **量子實現便宜** | RBS gate 是 2-qubit ops，shallow circuit |

### 2.2 RBS gate（Reconfigurable Beam Splitter）

兩-qubit gate，作用在 $\{|01\rangle, |10\rangle\}$ 子空間：

$$
\text{RBS}(\theta) = \begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & \cos\theta & \sin\theta & 0 \\
0 & -\sin\theta & \cos\theta & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}
$$

可用 **Hadamard + CZ + 單 qubit rotation** 分解（NISQ 友善）。

### 2.3 Pyramid Circuit

```
qubits  0 ─●─────●─────●─────●───
            │     │     │     │
qubits  1 ─●─────●─────●─────●───
                  │     │     │
qubits  2 ───────●─────●─────●───
                        │     │
qubits  3 ─────────────●─────●───
                              │
qubits  4 ───────────────────●───

       ↑ 每個 ● 是一個 RBS(θ_k)，總共 n(n-1)/2 個
```

對 $n=8$ qubits：**28 個 RBS gates → 28 個訓練參數**。

### 2.4 Unary Amplitude Encoding

把 $n$-維輸入 $x$ 編碼到 $n$ 個 qubit 的 unary state：
$$
|x\rangle = \sum_{i=0}^{n-1} x_i |0\cdots 1_i \cdots 0\rangle
$$

只用 **$n-1$ 個 RBS gate** 完成 encoding（比 amplitude encoding 的 $O(2^n)$ depth 便宜很多）。

---

## 3. Quantum Attention 機制

### 3.1 Classical attention 回顧

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

每個 entry $A_{ij} \propto \langle q_i, k_j\rangle$（內積）。

### 3.2 Quantum attention 替代

每個 attention coefficient 計算成：
$$
A_{ij} = x_i^T W x_j
$$

其中 $W$ 是 QONN 學出來的**正交矩陣**。具體量子流程：

```
Step 1: Load query x_i via unary encoding (n-1 RBS gates)
        |0...0⟩ → |x_i⟩

Step 2: Apply orthogonal transformation W
        |x_i⟩ → W|x_i⟩  (pyramid RBS circuit)

Step 3: Inverse-load key x_j (adjoint encoding)

Step 4: Measure probability of |1⟩ on first qubit
        P(|1⟩_0) ≈ |x_j^T W x_i|² = A_{ij}²
```

**關鍵差異**：取代 softmax 為 quantum measurement。

### 3.3 插入位置

```
┌──────────────────────────────────┐
│  Classical ViT Pipeline:         │
│  Patch → embed → Q,K,V proj      │
│         ↓                         │
│  ┌─────────────────────┐         │
│  │ ⚛ QUANTUM ATTENTION │  ← 唯一量子部分
│  │ A_ij = x_i^T W x_j  │         │
│  └─────────────────────┘         │
│         ↓                         │
│  Softmax (classical) → × V       │
│         ↓                         │
│  Output proj → MLP → next block  │
└──────────────────────────────────┘
```

不在 Q/K/V projection、不在 output projection——**只在 attention coefficient 計算這一步**。

---

## 4. 實驗設置

### 4.1 資料：CMS Open Data Jet Images

| 屬性 | 規格 |
|---|---|
| 總影像數 | 933,206 (sample 出 50,000) |
| 影像大小 | **125×125 pixels** |
| Channels | 3（Tracks / ECAL / HCAL）|
| 任務 | quark-initiated vs gluon-initiated jet 二元分類 |
| Auxiliary features | $p_T$, $m_0$（min-max scaled to [0,1]）|

### 4.2 Patch 配置

```
125×125 image → 25×25 patches → 25 patches per image
                                     ↓
                              projection dim = 8 → 8 qubits
```

### 4.3 訓練

| 參數 | 值 |
|---|---|
| Optimizer | Adam |
| Learning rate | 5e-4 |
| Epochs | 15 |
| Batch | 32 |
| Loss | Binary cross-entropy |
| Dropout | 0.5 |
| Transformer blocks | **僅 1 個 encoder block**（小規模） |
| Attention heads | 1 |
| Hardware | Quantum simulator（未明說）|

**作者自承計算限制**：`"the simulation requires 26×26 attention circuits per self-attention block, which already pushes the computational limits of current quantum simulators."`

### 4.4 切分

70% train / 15% val / 15% test，從 50,000 subset 抽

---

## 5. 量化結果

### 5.1 Test set 表現

| Model | Test Loss | Test Accuracy | Test AUC |
|---|---|---|---|
| **Classical ViT** | 0.6087 | 67.88% | **0.7385** |
| **Quantum QViT** | 0.6105 | 67.55% | **0.7369** |

**差距：AUC 差 0.0016（< 0.3%），accuracy 差 0.33%**——**幾乎完全平手**。

### 5.2 為什麼「平手」其實是好結果？

| 觀點 | 解讀 |
|---|---|
| ❌ 悲觀 | quantum 沒帶來優勢，理論預期沒實現 |
| ✅ 樂觀 | **僅用 28 個量子參數 match 了 classical 完整 attention**（雖然沒給 classical 參數量）|
| ✅ 工程 | 證明 quantum attention 可訓、不崩、可 reproduce |
| ✅ Framing | "**robust performance + promising scalability**"——這正是 GSoC proposal 應該採用的措辭 |

---

## 6. 與你 GSoC 的 5 個直接連結

### 連結 1：Sergei Gleyzer 共同作者

- 他是 ML4SCI 的 PI、DeepLense 的主理人
- 你 paper 1 (Alexander 2019) + paper 2 (Alexander 2021) 的 author
- **這篇是 lensing 的 sister project (HEP)，用同樣 quantum philosophy**
- 提案時應引用：「本工作延續 Gleyzer 等於 HEP 領域 (Tesi 2024) 的 quantum attention 方向，首次應用於 lensing」

### 連結 2：QONN attention 可直接套到 mae-lensing

`mainv2.py` 的 `TransformerBlock` (line 394)：

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ...):
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, ...)
        # ← 替換成 QuantumOrthogonalAttention
```

**插入點明確**：6 個 encoder block，可選擇全替換或只替換最後一個。

### 連結 3：對比 paper 3 (Pasquali 2024 QViT) 的進化

| 維度 | Pasquali 2024 QViT | **Tesi 2024 QONN-ViT** |
|---|---|---|
| Attention 替換策略 | Naive quantum self-attention | **正交約束的 quantum attention** |
| 量子 gate 設計 | 一般 PQC | **RBS pyramid (專為 orthogonal)** |
| Encoding | Amplitude (深) | **Unary (淺)** |
| Parameter efficiency | 較差 | **$n(n-1)/2$ 剛好對齊正交矩陣自由度** |
| 結果 | competitive | competitive |

**Tesi 是 Pasquali 的進化版**——你 proposal 應該優先引 Tesi 而非 Pasquali。

### 連結 4：3 種架構選擇（給你 D3 升級）

把 Tesi 整合進 mae-lensing，得到三種選擇：

| 變體 | Encoder | Attention | Classifier head | Novelty |
|---|---|---|---|---|
| **D3-a (你原本)** | Classical ViT (frozen) | Classical | **QuantumFusionHead** | 中 |
| **D3-b (QONN attention)** | Classical ViT with QONN attention | **Quantum (QONN)** | Classical Linear | 高 |
| **D3-c (full quantum)** | ViT-QONN encoder | **Quantum** | **QuantumFusionHead** | 極高（但風險大）|

**建議**：**D3-b 是甜蜜點**——只動 attention 一層，其他保持 mae-lensing 原樣，可重用預訓 weights 的大部分。

### 連結 5：Proposal 寫作的 framing template

Tesi 結論的寫法（**幾乎完全平手卻仍正面**）：

> "The results indicate that embedding quantum orthogonal transformations within the attention mechanism can provide **robust performance** while offering **promising scalability** for machine learning challenges associated with the upcoming High Luminosity Large Hadron Collider."

你的 proposal 可仿照：

> "本工作將 quantum orthogonal attention 整合進 strong-lensing MAE pipeline，預期結果不在 raw accuracy 突破當前 SOTA (AUC 0.968)，而在於 **(i) 證明 quantum-enhanced architecture 可在 lensing 任務上達到 comparable performance with parameter efficiency, (ii) 為 LSST 時代 10⁵-lens scale analysis 提供 quantum scalability path, (iii) 揭示 orthogonal transformation 在 substructure feature learning 的 representational advantage。**"

---

## 7. 對你 D3 (Quantum MAE + Fusion) 的具體影響

### 修正版 D3 架構（吸收 Tesi 2024）

```
═══════════════════════════════════════════════════════════════
Phase 1: MAE Pretraining (沿用 mae-lensing 原 code)
  Classical ViT-Tiny + standard self-attention
  Pretrain on no_sub class only
  → mae_encoder.pth  (≈ AUC 0.968 ceiling)

═══════════════════════════════════════════════════════════════
Phase 2: Quantum Fine-tuning 三種 variants
═══════════════════════════════════════════════════════════════

Variant A (你原本): QuantumFusionHead
  └─ Frozen encoder + Cross-attn + TSHF + Linear classifier
  └─ Quantum 只在 head

Variant B (新加, Tesi-inspired): QONN Attention
  └─ Replace last encoder block's attention with QONN (8 qubits, pyramid)
  └─ Frozen 前 5 blocks，fine-tune 第 6 block + head
  └─ Quantum 在 attention 內部

Variant C (究極): QONN Attention + QuantumFusionHead
  └─ Two-place quantum
  └─ 需要 ablation 區分兩處貢獻
```

**推薦執行順序**：A → B → C（漸進加深）。

---

## 8. PennyLane / Qiskit 實作要點

### 8.1 RBS gate 在 PennyLane 中

PennyLane 沒內建 RBS，但可用 `qml.SingleExcitation` 等價：

```python
import pennylane as qml

def RBS(theta, wires):
    """Reconfigurable Beam Splitter on two qubits."""
    # Equivalent to qml.SingleExcitation up to phase
    qml.SingleExcitation(theta, wires=wires)
```

或手動分解：
```python
def RBS(theta, wires):
    qml.Hadamard(wires[0])
    qml.Hadamard(wires[1])
    qml.CZ(wires=wires)
    qml.RY(theta / 2, wires=wires[0])
    qml.RY(-theta / 2, wires=wires[1])
    qml.CZ(wires=wires)
    qml.Hadamard(wires[0])
    qml.Hadamard(wires[1])
```

### 8.2 Pyramid circuit 範例（8 qubits）

```python
def pyramid_QONN(thetas, n_qubits=8):
    """Pyramid of n(n-1)/2 = 28 RBS gates for 8x8 orthogonal matrix."""
    idx = 0
    for layer in range(n_qubits - 1):
        for i in range(n_qubits - 1 - layer):
            RBS(thetas[idx], wires=[i, i + 1])
            idx += 1
```

### 8.3 Unary loading（向量編碼）

```python
def load_vector(x, n_qubits):
    """Encode normalized n-d vector into unary state |0...1_i...0⟩."""
    qml.PauliX(wires=0)  # Initial state |10...0⟩
    # Angles from Givens decomposition of x
    angles = compute_givens_angles(x)
    for i, theta in enumerate(angles):
        RBS(theta, wires=[i, i + 1])
```

---

## 9. 已知限制

| 限制 | 影響 |
|---|---|
| **僅 1 個 encoder block, 1 head** | 比 mae-lensing (6 blocks) 小很多——直接搬不來，要 scale up |
| **AUC 0.737 非常低** | quark/gluon jet 本來就難分（標準 ML 也僅 ~0.85）；不要拿這數字嚇到 |
| **沒給 parameter count** | 無法做嚴格 parameter-matched comparison |
| **沒跟 El Cherrat 2024 / Guo 2024 比** | 引用了但沒實驗對比 |
| **Simulator 計算瓶頸** | 26×26 attention circuits/block 已是極限——對 mae-lensing 256×256 = 4096 attention 計算更難 |

> **對 mae-lensing 移植的 implication**：你**不能直接套 QONN attention 到所有 256 tokens**。可行做法：(a) 只在 CLS token 與 patch tokens 之間做 QONN attention，(b) 只在 final block 做，(c) 降到 8×8 patches 減少 token 數。

---

## 10. 候選 BibTeX

```bibtex
@article{tesi2024quantum,
  title={Quantum Attention for Vision Transformers in High Energy Physics},
  author={Tesi, Alessandro and Dahale, Gopal Ramesh and Gleyzer, Sergei
          and Kong, Kyoungchul and Magorsch, Tom and
          Matchev, Konstantin T. and Matcheva, Katia},
  year={2024},
  archivePrefix={arXiv},
  eprint={2411.13520},
  primaryClass={hep-ph}
}
```

---

## 11. 一句話總結

> **Tesi 2024 (含 Gleyzer 共同作者) 用 RBS pyramid 實作 quantum orthogonal attention**，**取代 ViT 標準 attention 的內積運算**，**結果與 classical ViT 平手（AUC 0.737 vs 0.739）**。這篇是 paper 3 (Pasquali 2024 QViT) 的**正交化進化版**，**Sergei Gleyzer 同時是 DeepLense 主理人**——你的 GSoC proposal 應該**優先引用此篇**，把 D3 升級為「**MAE encoder + QONN attention final block + Quantum Fusion head**」的三層整合架構。

---

## 12. 更新 04_GSoC_QML_Proposal.md 的建議

### Related Work 段加入

```
量子注意力機制方向，Tesi et al. 2024 (含 Gleyzer，與 DeepLense 同團隊) 提出
quantum orthogonal neural network (QONN) attention，使用 RBS pyramid 在 jet
classification 達到與 classical ViT 持平表現 (AUC 0.737 vs 0.739)。
本工作將此設計從 jet (HEP) 移植至 lensing (astro)，並與 MAE pretraining 整合。
```

### Methodology D3 加入 Variant B

新增 Stage 2 quantum option：
- **Q-Attn block**：替換 ViT 第 6 block 的 attention 為 QONN-pyramid（8 qubits, 28 params）
- Frozen 前 5 blocks，fine-tune Q-Attn + head
- 跟 Variant A (QuantumFusionHead) 做 ablation

---

## 13. 待跟進

- [ ] 讀 Tesi 2024 PDF 的 Section 3-4 (full method details with figures)
- [ ] 找 El Cherrat 2024（QViT 先驅）對比閱讀
- [ ] 把 D3 Variant B 整合進 `mainv2.py` 的 `TransformerBlock`
- [ ] 在 04_GSoC_QML_Proposal.md v2 更新引用此篇

---

> **VSCode 預覽**：`Ctrl+Shift+V`
