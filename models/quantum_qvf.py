#!/usr/bin/env python
"""
QVF-style classifier for DeepLense dark-matter 3-class (axion/cdm/no_sub).

Adapts the key idea of QVF (Wang et al., NeurIPS 2025, arXiv:2508.10900):
NEURAL AMPLITUDE ENCODING grounded in a learnable energy manifold, replacing
the fixed amplitude embedding we showed is classically trivial. A learnable
energy net E_φ(x) defines a Boltzmann amplitude distribution
    |a_i|² = softmax(−E_φ(x))_i ,   a_i = √(|a_i|²)
which is amplitude-embedded into N_Q qubits, then a fully-entangled PQC U(θ),
then ⟨Z⟩ → classical head → logits. Everything (φ, θ, head) trains end-to-end.

  --sham : same learnable NAE but a classical Linear replaces the circuit
           → isolates whether the QUANTUM circuit adds anything over the
           (classical) learnable encoding.

Backend: PennyLane default.qubit + backprop (same SLURM/PennyLane pipeline).
"""
import os, sys
if "jax" not in sys.modules:
    sys.modules["jax"] = None
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

N_Q = 8
DIM = 2 ** N_Q                                  # 256 amplitudes
K_LATENT = int(os.environ.get("QVF_K", "8"))
N_LAYERS = int(os.environ.get("QVF_LAYERS", "4"))
ENERGY_HID = int(os.environ.get("QVF_HID", "128"))
_BACKEND = f"QVF default.qubit (N_Q={N_Q}, K={K_LATENT}, layers={N_LAYERS}, e_hid={ENERGY_HID})"

_DEV = qml.device("default.qubit", wires=N_Q)


def enc_shape():
    return qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_Q)


@qml.qnode(_DEV, interface="torch", diff_method="backprop")
def _circuit(amp, weights):
    qml.AmplitudeEmbedding(amp, wires=range(N_Q), normalize=True)
    qml.StronglyEntanglingLayers(weights, wires=range(N_Q))     # fully-entangled PQC
    return [qml.expval(qml.PauliZ(i)) for i in range(K_LATENT)]


class NeuralAmplitudeEncoding(nn.Module):
    """Learnable energy manifold → Boltzmann amplitude state (QVF-style)."""

    def __init__(self, in_dim):
        super().__init__()
        self.energy = nn.Sequential(
            nn.Linear(in_dim, ENERGY_HID), nn.Tanh(),
            nn.Linear(ENERGY_HID, DIM))

    def forward(self, x):                       # (B, in_dim) → amplitudes (B, 256)
        e = self.energy(x)
        a2 = torch.softmax(-e, dim=1)           # |a_i|² = Boltzmann over learnable energy
        return torch.sqrt(a2 + 1e-12)


class QVFClassifier(nn.Module):
    def __init__(self, in_dim=192, n_classes=3, sham=False):
        super().__init__()
        self.sham = sham
        self.nae = NeuralAmplitudeEncoding(in_dim)
        if sham:
            self.cl = nn.Linear(DIM, K_LATENT)
        else:
            self.w = nn.Parameter(0.1 * torch.randn(enc_shape()))
        self.head = nn.Sequential(nn.LayerNorm(K_LATENT), nn.Linear(K_LATENT, n_classes))

    def forward(self, x):
        amp = self.nae(x)                       # (B, 256) learnable amplitudes
        if self.sham:
            z = torch.tanh(self.cl(amp))
        else:
            z = torch.stack(_circuit(amp, self.w), dim=-1).to(amp.dtype)
        return self.head(z)
