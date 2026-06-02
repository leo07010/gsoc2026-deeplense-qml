#!/usr/bin/env python
"""
F2 gated-residual quantum-classical fusion head — PennyLane lightning.gpu backend.

SAME model as quantum_fusion_cudaq.py (16-qubit dressed PQC, RY encode once →
[RZ,RY + CNOT-ring] × N_LAYERS → ⟨Z_i⟩, gated-residual fusion with a classical
linear head). The ONLY difference is the gradient engine:

    cudaq version : hand-written parameter-shift   → O(2·P) circuit evals/step
    THIS  version : adjoint differentiation         → O(1)   circuit evals/step

On a state-vector simulator adjoint is the standard, order-of-magnitude-faster
choice (it back-propagates through the simulation instead of re-running the
circuit twice per parameter). At init gate=0 ⇒ model == classical baseline.

ENV NOTE: PennyLane 0.42 + JAX 0.9 are import-incompatible (JAX dropped
`DynamicJaxprTrace`). lightning.gpu + the torch interface never need JAX, so we
hide it before importing PennyLane to take the no-capture import path. This
touches nothing global.
"""
import sys
if "jax" not in sys.modules:               # force PennyLane's "jax unavailable" path
    sys.modules["jax"] = None

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

# ── Circuit size knobs (identical to quantum_fusion_cudaq.py) ──────────
N_Q = 16
N_LAYERS = 4
N_WEIGHTS = N_Q * 2 * N_LAYERS     # RZ + RY per qubit per layer = 128

# Backend choice (benchmarked on H100, 16 qubits, batch of 64):
#   default.qubit + backprop  : 0.33 s/batch   ← fully batched torch on GPU (WIN)
#   lightning.gpu + adjoint   : 18   s/batch   (per-sample loop, no broadcast)
#   cudaq parameter-shift     : 57   s/batch   (quantum_fusion_cudaq.py)
# default.qubit runs the whole state vector as a torch tensor with the batch as
# a leading dim, so one call differentiates the entire batch via torch autograd
# (backprop) — exact, GPU-resident, ~170x faster than parameter-shift here.
_DEV = qml.device("default.qubit", wires=N_Q)
_BACKEND = "default.qubit+backprop"


@qml.qnode(_DEV, diff_method="backprop", interface="torch")
def _circuit(angles, weights):
    """angles: (B, N_Q) broadcast ; weights: (N_WEIGHTS,). Returns list of N_Q ⟨Z_i⟩,
    each shaped (B,) — torch autograd connects them straight back to angles/weights."""
    for i in range(N_Q):
        qml.RY(angles[..., i], wires=i)                  # encode once (broadcast over batch)
    for L in range(N_LAYERS):
        base = L * N_Q * 2
        for i in range(N_Q):
            qml.RZ(weights[base + i * 2], wires=i)
            qml.RY(weights[base + i * 2 + 1], wires=i)
        for i in range(N_Q - 1):
            qml.CNOT(wires=[i, i + 1])                    # CNOT ring
        qml.CNOT(wires=[N_Q - 1, 0])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_Q)]


def _run_batch(angles, weights):
    """(B, N_Q) angles + (N_WEIGHTS,) weights → (B, N_Q) ⟨Z⟩, fully batched."""
    return torch.stack(_circuit(angles, weights), dim=-1)   # (B, N_Q)


class QuantumFusionHead(nn.Module):
    """Gated residual fusion: classical linear head + adjoint-trained quantum branch."""

    def __init__(self, in_dim=192, n_classes=3):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, N_Q))
        self.qweights = nn.Parameter(0.01 * torch.randn(N_WEIGHTS))
        self.q_out = nn.Linear(N_Q, n_classes)
        self.classical = nn.Linear(in_dim, n_classes)
        self.gate = nn.Parameter(torch.zeros(n_classes))   # tanh(0)=0 ⇒ baseline at init

    @torch.no_grad()
    def init_classical(self, head_weight, head_bias):
        self.classical.weight.copy_(torch.as_tensor(head_weight))
        self.classical.bias.copy_(torch.as_tensor(head_bias))

    def forward(self, cls_feat):
        angles = torch.tanh(self.proj(cls_feat)) * np.pi          # bound to [-π, π]
        qz = _run_batch(angles, self.qweights)                    # (B, N_Q)
        logits_q = self.q_out(qz.to(cls_feat.dtype))
        logits_c = self.classical(cls_feat)
        return logits_c + torch.tanh(self.gate) * logits_q
