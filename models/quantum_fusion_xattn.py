#!/usr/bin/env python
"""
Cross-Attention Mid-Fusion head (token-level quantum-classical fusion).

Re-implements the mid-fusion of
  Alavi, Kouchmeshki & Alavi, "Practical Quantum-Classical Feature Fusion for
  complex data Classification", arXiv:2512.19180 (Dec 2025),
adapted to DeepLense: the frozen-ViT 192-d CLS feature is the classical CLS
token, the 16-qubit PQC emits 2Q = 32 quantum readout tokens (local ⟨Z_j⟩ +
ring ⟨Z_jZ_{j+1}⟩), and a Transformer self-attention block lets the classical
CLS token attend over the quantum tokens. The fused CLS is read out to class
logits.

Difference from the paper (kept from our F2 head): the fusion logits are added
to a frozen classical-baseline head through a zero-initialised gate, so at init
the model is EXACTLY the classical baseline (AUC 0.9734) and can only improve.

Backend: PennyLane default.qubit + backprop (batched torch on GPU) — same fast
engine as quantum_fusion_pennylane.py (~170x over cudaq parameter-shift).
"""
import os
import sys
if "jax" not in sys.modules:
    sys.modules["jax"] = None

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

# Variants (set via env before import; train_fusion_pennylane.py wires --reupload/--pure):
#   QF_REUPLOAD=1 : data re-uploading — re-encode the input angles at EVERY layer
#                   (raises per-qubit expressivity, eases the 192->16 bottleneck).
#   QF_PURE=1     : paper-faithful pure mid-fusion — read the fused CLS straight to
#                   logits (NO gated residual onto the frozen classical baseline).
_REUPLOAD = os.environ.get("QF_REUPLOAD", "0") == "1"
_PURE = os.environ.get("QF_PURE", "0") == "1"
# QF_SHAM=1 → ablation: replace the 32 quantum readout tokens with a classical
# projection of the CLS feature (same token count/dim, NO circuit). Quantum vs
# sham under identical architecture is the only clean test of quantum advantage.
_SHAM = os.environ.get("QF_SHAM", "0") == "1"

# ── circuit / fusion size knobs ───────────────────────────────────────
N_Q = 16
N_LAYERS = 4
N_WEIGHTS = N_Q * 2 * N_LAYERS        # RZ + RY per qubit per layer = 128
N_TOK = 2 * N_Q                       # 16 local ⟨Z⟩ + 16 ring ⟨ZZ⟩ = 32 tokens
D_TOK = 64                            # token embedding dim
N_HEADS = 4

_DEV = qml.device("default.qubit", wires=N_Q)
_BACKEND = (f"{'SHAM-classical' if _SHAM else 'default.qubit+backprop'} (xattn"
            f"{'+reupload' if _REUPLOAD and not _SHAM else ''}{'+pure' if _PURE else '+gated'})")


@qml.qnode(_DEV, diff_method="backprop", interface="torch")
def _circuit(angles, weights):
    """angles (B, N_Q) broadcast, weights (N_WEIGHTS,).
    Returns 2Q expvals: ⟨Z_j⟩ for j, then ring ⟨Z_j Z_{j+1}⟩.
    With re-uploading the input angles are re-encoded before every layer."""
    if not _REUPLOAD:
        for i in range(N_Q):
            qml.RY(angles[..., i], wires=i)               # encode once
    for L in range(N_LAYERS):
        if _REUPLOAD:
            for i in range(N_Q):
                qml.RY(angles[..., i], wires=i)           # re-upload data each layer
        base = L * N_Q * 2
        for i in range(N_Q):
            qml.RZ(weights[base + i * 2], wires=i)
            qml.RY(weights[base + i * 2 + 1], wires=i)
        for i in range(N_Q - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_Q - 1, 0])
    local = [qml.expval(qml.PauliZ(j)) for j in range(N_Q)]
    ring = [qml.expval(qml.PauliZ(j) @ qml.PauliZ((j + 1) % N_Q)) for j in range(N_Q)]
    return local + ring


def _quantum_tokens(angles, weights):
    """(B, N_Q) angles → (B, N_TOK) quantum readout scalars (fully batched)."""
    return torch.stack(_circuit(angles, weights), dim=-1)      # (B, 2Q)


class QuantumFusionHead(nn.Module):
    """Token-level cross-attention fusion, gated onto the classical baseline."""

    def __init__(self, in_dim=192, n_classes=3):
        super().__init__()
        # ── quantum branch → tokens ──
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, N_Q))
        self.scale = nn.Parameter(torch.ones(N_Q))          # trainable per-wire angle scale
        self.qweights = nn.Parameter(0.01 * torch.randn(N_WEIGHTS))
        if _SHAM:
            self.sham_proj = nn.Linear(in_dim, N_TOK)       # classical stand-in for the circuit
        self.tok_embed = nn.Linear(1, D_TOK)                # shared scalar→D_TOK embed
        self.tok_id = nn.Parameter(0.02 * torch.randn(N_TOK, D_TOK))  # identity embeddings

        # ── classical CLS token ──
        self.cls_proj = nn.Linear(in_dim, D_TOK)

        # ── transformer self-attention block (pre-LN, 4 heads, FFN×4) ──
        self.attn = nn.TransformerEncoderLayer(
            d_model=D_TOK, nhead=N_HEADS, dim_feedforward=4 * D_TOK,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.fusion_out = nn.Linear(D_TOK, n_classes)        # read fused CLS → logits

        # ── classical baseline branch (init from shipped head) + zero gate ──
        self.classical = nn.Linear(in_dim, n_classes)
        self.gate = nn.Parameter(torch.zeros(n_classes))

    @torch.no_grad()
    def init_classical(self, head_weight, head_bias):
        self.classical.weight.copy_(torch.as_tensor(head_weight))
        self.classical.bias.copy_(torch.as_tensor(head_bias))

    def forward(self, cls_feat):
        B = cls_feat.shape[0]
        # quantum tokens (or sham classical stand-in)
        if _SHAM:
            qz = torch.tanh(self.sham_proj(cls_feat))                   # (B, N_TOK) classical
        else:
            angles = torch.tanh(self.scale * self.proj(cls_feat)) * np.pi
            qz = _quantum_tokens(angles, self.qweights).to(cls_feat.dtype)
        q_tok = self.tok_embed(qz.unsqueeze(-1)) + self.tok_id          # (B, N_TOK, D_TOK)
        # classical CLS token
        cls_tok = self.cls_proj(cls_feat).unsqueeze(1)                  # (B, 1, D_TOK)
        # sequence [CLS ; quantum tokens] → self-attention → read CLS
        seq = torch.cat([cls_tok, q_tok], dim=1)                        # (B, 1+N_TOK, D_TOK)
        fused = self.attn(seq)[:, 0]                                    # (B, D_TOK)
        logits_fuse = self.fusion_out(fused)                           # (B, n_classes)
        if _PURE:
            # paper-faithful: fused CLS is read straight to logits (no baseline)
            return logits_fuse
        # gated residual onto frozen classical baseline (gate=0 ⇒ baseline at init)
        return self.classical(cls_feat) + torch.tanh(self.gate) * logits_fuse
