#!/usr/bin/env python
"""
Quantum-Classical Transformer (QCT) head — mixed-self-attention fusion of
classical ViT patch tokens with quantum readout tokens.

Realises the "Quantum Encoder + hybrid attention" design:
  classical : 256 frozen-ViT patch tokens  →  Linear(192→D)            ─┐
  quantum   : CLS(192) → angles → PQC → 2Q readout tokens → Linear(1→D) ─┤
  + learnable [CLS] token, + per-type embeddings                         │
  sequence  [CLS ; 256 patch ; 2Q quantum]  → L× Transformer self-attn ──┘
  read fused [CLS] → Linear(D→n_classes)

Both the quantum circuit (qweights) and the attention block train jointly via
backprop (default.qubit). The classical patch tokens and quantum tokens are
fused *as equals* inside one self-attention stack — the model learns which
patch/quantum tokens to attend to.

Env flag QF_REUPLOAD=1 → data re-uploading in the circuit.
"""
import os, sys
if "jax" not in sys.modules:
    sys.modules["jax"] = None
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

_REUPLOAD = os.environ.get("QF_REUPLOAD", "0") == "1"
# QF_SHAM=1 → ablation control: replace the quantum readout tokens with a purely
# classical projection of the CLS feature (same token count/dim, NO circuit).
# If sham ≈ quantum, the gain is the transformer's, not the quantum branch's.
_SHAM = os.environ.get("QF_SHAM", "0") == "1"

# ── sizes ─────────────────────────────────────────────────────────────
N_Q = 16
N_LAYERS = 4                      # PQC entangling layers
N_WEIGHTS = N_Q * 2 * N_LAYERS    # 128
N_TOK = 2 * N_Q                   # 32 quantum readout tokens (Z + ring ZZ)
N_PATCH = 256                     # ViT patch tokens
D = 128                           # transformer model dim
N_HEADS = 4
N_TBLOCKS = 2                     # transformer depth

_DEV = qml.device("default.qubit", wires=N_Q)
_BACKEND = (f"QCT {'SHAM-classical' if _SHAM else 'default.qubit+backprop'}"
            f"{'+reupload' if _REUPLOAD and not _SHAM else ''}")


@qml.qnode(_DEV, diff_method="backprop", interface="torch")
def _circuit(angles, weights):
    if not _REUPLOAD:
        for i in range(N_Q):
            qml.RY(angles[..., i], wires=i)
    for L in range(N_LAYERS):
        if _REUPLOAD:
            for i in range(N_Q):
                qml.RY(angles[..., i], wires=i)
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
    return torch.stack(_circuit(angles, weights), dim=-1)      # (B, N_TOK)


class QuantumFusionHead(nn.Module):
    """QCT head. forward(patches, cls_feat) -> logits."""

    def __init__(self, in_dim=192, n_classes=3):
        super().__init__()
        # classical patch tokens
        self.patch_proj = nn.Linear(in_dim, D)
        # quantum branch (or sham classical token generator)
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, N_Q))
        self.scale = nn.Parameter(torch.ones(N_Q))
        self.qweights = nn.Parameter(0.01 * torch.randn(N_WEIGHTS))
        if _SHAM:
            self.sham_proj = nn.Linear(in_dim, N_TOK)   # classical stand-in for the circuit
        self.tok_embed = nn.Linear(1, D)
        self.tok_id = nn.Parameter(0.02 * torch.randn(N_TOK, D))
        # learnable CLS + per-type embeddings (cls / patch / quantum)
        self.cls_token = nn.Parameter(0.02 * torch.randn(1, 1, D))
        self.type_emb = nn.Parameter(0.02 * torch.randn(3, D))   # [cls, patch, quantum]
        # mixed self-attention stack
        block = nn.TransformerEncoderLayer(
            d_model=D, nhead=N_HEADS, dim_feedforward=4 * D,
            dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, num_layers=N_TBLOCKS)
        self.norm = nn.LayerNorm(D)
        self.out = nn.Linear(D, n_classes)

    # kept for interface-compatibility with the other heads (no-op here)
    @torch.no_grad()
    def init_classical(self, head_weight, head_bias):
        pass

    def forward(self, patches, cls_feat):
        B = cls_feat.shape[0]
        # classical patch tokens
        p_tok = self.patch_proj(patches) + self.type_emb[1]            # (B, 256, D)
        # quantum tokens (or sham classical stand-in)
        if _SHAM:
            qz = torch.tanh(self.sham_proj(cls_feat))                  # (B, N_TOK) classical
        else:
            angles = torch.tanh(self.scale * self.proj(cls_feat)) * np.pi
            qz = _quantum_tokens(angles, self.qweights).to(cls_feat.dtype)
        q_tok = self.tok_embed(qz.unsqueeze(-1)) + self.tok_id + self.type_emb[2]
        # CLS token
        cls = self.cls_token.expand(B, -1, -1) + self.type_emb[0]      # (B, 1, D)
        # mixed sequence -> self-attention -> read CLS
        seq = torch.cat([cls, p_tok, q_tok], dim=1)                    # (B, 1+256+N_TOK, D)
        fused = self.norm(self.encoder(seq)[:, 0])                     # (B, D)
        return self.out(fused)
