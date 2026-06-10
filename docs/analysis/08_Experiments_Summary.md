# 08 — 實驗總覽：測試過哪些方法、得到什麼效果

> **作者**：[@leo07010](https://github.com/leo07010)
> **整理日期**：2026-06-02
> **目的**：彙整本專案在 DeepLense × QML 上實作與測試過的所有方法，及各自的結果與證據強度。
> **相關文件**：方法設計見 [`04_GSoC_QML_Proposal.md`](04_GSoC_QML_Proposal.md)、[`07_QMAE_DeepLense_Design.md`](07_QMAE_DeepLense_Design.md)；上游 repo 拆解見 [`../02_Code/mae-lensing/REPO_ANALYSIS.md`](../02_Code/mae-lensing/REPO_ANALYSIS.md)。

---

## 一句話總結

以 **2025 年 DeepLense MAE SOTA（古典 ViT，AUC 0.968）為底**，系統性測試「把量子模組接上去能不能贏、有沒有用」。方法上的關鍵設計是**每個量子方法都配一個 capacity-matched 的 sham（假量子）對照組**。結論誠實：**在強古典 baseline + 海量資料的判別任務上，量子沒有可測量的增益（quantum ≈ sham）**，因此方向**從「判別」轉向「生成／自監督重建」（QMAE）**作為真正的 novelty。

**證據圖例**：✅ 有實測數字　｜　🟡 只有質性結論／註記參考值　｜　⏳ 程式碼完成但結果尚未回填

---

## 表 1 — 古典 baseline 與分析（第 0 階段）

| # | 方法 | 檔案 | 目的 | 結果 | 證據 |
|---|---|---|---|---|---|
| 0a | MAE pretrained classifier 評估 | `eval_pretrained.py` | 量出要打的目標 | **AUC 0.974 / acc 88.65%** | ✅ |
| 0b | 逐樣本誤差分析（8910 val） | `analyze_errors.py` + `error_analysis/` | 找古典模型弱點 | **acc 0.8847；axion recall 0.769、cdm 0.955、no_sub 0.928；主混淆 axion→cdm（677 張）** | ✅ |
| 0c | 特徵抽取 | `extract_features.py`、`extract_patch_features.py`、`extract_images16.py` | 為量子實驗 cache 192-d CLS／256 patch token／16×16 影像 | 產出可重用 cache | ✅ |
| 0d | t-SNE 視覺化 | `enhanced_tsne.py` | 看 representation 分離度 | axion/cdm 群重疊（對應 0b） | ✅ |

**這一步的價值**：確認了「要打的目標」(AUC 0.974) 與「古典模型的弱點」(axion recall 0.77)，後續所有量子實驗共用同一份 frozen encoder 特徵，確保公平對比。

---

## 表 2 — 量子「判別」實驗（第 1 階段，接 frozen MAE encoder）

全部採 **gate 初值 = 0 ⇒ 起點等於 baseline AUC 0.974，只能往上**的設計；全部配 `--sham`（同容量古典假量子）對照。

| # | 方法 | 檔案 | 量子設計 | 結果 | 證據 |
|---|---|---|---|---|---|
| 1 | **F2 — Gated Residual Fusion** | `quantum_fusion_cudaq.py`、`quantum_fusion_pennylane.py` | 16-qubit dressed PQC（RY 編碼一次 → [RZ,RY + CNOT-ring]×4 → ⟨Z⟩），與古典 linear head gated 殘差融合 | 納入「quantum ≈ sham」總結 | 🟡 |
| 2 | **Cross-Attention Mid-Fusion** | `quantum_fusion_xattn.py` | 復刻 Alavi et al. 2512.19180：32 量子讀出 token + CLS token 過 self-attention（變體 `reupload`/`pure`/`sham`） | 納入「quantum ≈ sham」總結 | 🟡 |
| 3 | **QCT — Quantum-Classical Transformer** | `quantum_fusion_qct.py` + `train_qct.py` | token-level 全融合：256 ViT patch token + 32 量子 token + CLS 進 mixed self-attention | **明確記錄：QCT quantum = QCT sham** | 🟡 |

### 第 1 階段總結論（[`07_QMAE_DeepLense_Design.md`](07_QMAE_DeepLense_Design.md)）

> 多組實驗（gated / xattn / QCT + sham 對照）顯示：**在強古典 baseline + 海量資料下，量子在判別任務上沒有可測量的增益（QCT quantum = QCT sham）**。增益（若有）來自融合架構本身，而非量子電路。

這正是 proposal 預先 frame 的「characterize where it helps, not assume it wins」——一個有價值的 negative result。

### 附帶的工程結果（docstring 實測）

梯度引擎 benchmark（16 qubit、batch 64、H100）：

| 引擎 | 速度 |
|---|---|
| CUDA-Q parameter-shift | 57 s/batch |
| lightning.gpu + adjoint | 18 s/batch |
| **`default.qubit` + backprop（全 batch torch on GPU）** | **0.33 s/batch（快 ~170×）** |

→ 後續實驗全改用 `default.qubit + backprop`。
另註記參考值：axion recall — baseline 0.754 → 古典-only 重訓 0.788（量子線未超越）。

---

## 表 3 — 量子「生成／自監督」實驗（第 2 階段，pivot 後的真 novelty）

| # | 方法 | 檔案 | 設計 | 對標 | 結果 | 證據 |
|---|---|---|---|---|---|---|
| 4 | **QMAE（量子遮罩自編碼器）** | `quantum_mae.py`、`train_qmae.py`、`train_qmae_cls.py` | 忠實復刻 Andrews et al. 2511.17372：16×16 → amplitude embed 8 qubit → U(θ) → SWAP latent（trash 天然重置）→ U†(θ) → 重建 fidelity（自監督，無標籤）；下游 latent ⟨Z⟩ 三分類 + sham | 量子 baseline + sham（**非** 0.968） | 尚未回填 | ⏳ |
| 5 | **Quantum AE 異常偵測** | `train_qae_anomaly.py`、`train_qae_anomaly_cls.py` | Romero 2017 trash-qubit 壓縮，只訓 no_sub，重建 fidelity = 異常分數；輸入 16×16 與 192-d CLS 兩版 | Alexander 2021 AAE ≈ 0.93 | 尚未回填 | ⏳ |
| 6 | **Equivariant 量子殘差 + few-shot 掃描** | `quantum_equiv.py`、`train_fewshot.py` | C4 群平均 → 旋轉不變量子特徵；N=25/50/100/250/500 per-class 比 classical / sham / quantum | sham（小樣本 regime） | 尚未回填 | ⏳ |

**定位（doc 07）**：proof-of-concept，對標「量子 baseline + 同容量古典 sham」，**不是要贏古典 MAE 的 0.968**（QMAE 原文在 MNIST 也僅 65%）。貢獻 = 第一個把 QMAE 套到強透鏡暗物質資料 + 嚴謹 sham 對照 + few-shot 分析。

---

## 一頁總覽

| 階段 | 測了幾種方法 | 核心發現 | 證據強度 |
|---|---|---|---|
| 0 古典 baseline + 分析 | 4 | AUC 0.974；瓶頸是 axion recall 0.77（axion↔cdm 混淆） | ✅ 有實測 |
| 1 量子判別（3 架構 ×{quantum, sham, 變體}） | 3 | **quantum ≈ sham，無可測量量子增益** | 🟡 質性結論 |
| 2 量子生成/SSL | 3 | 程式碼齊備，定位 proof-of-concept | ⏳ 未跑完 |

---

## 結果存檔現況（誠實聲明）

- **僅有古典 baseline 的數字有實際存檔**：`02_Code/mae-lensing/error_analysis/`（含 `error_analysis.csv` 與圖）。
- **量子線（表 2、表 3）目前無逐項數字存檔**：無 `outputs_*/` 目錄、無訓練 log、無結果 CSV。
  - 表 2 依據：doc 07 的文字結論 + 各 `quantum_fusion_*.py` 的 docstring 參考值。
  - 表 3 三條線：程式碼完成，里程碑仍標 ⏳（待跑）。

---

## 建議下一步

1. **回填表 2 的實際數字**——產出 `classical / sham / quantum × {gated, xattn, QCT}` 的 AUC 對照表，這是 negative result 最有說服力的證據。
2. **跑完表 3 三條生成線**，尤其 `train_fewshot.py` 的小樣本曲線（量子最可能贏 sham 的 regime）。
3. **加 results-logger**：把每次訓練 stdout 結構化存成 CSV/JSON（套既有 `error_analysis/` 格式），避免再次只留結論不留數字。
