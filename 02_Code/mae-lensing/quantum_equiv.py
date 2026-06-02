#!/usr/bin/env python
"""
Equivariant quantum residual head for DeepLense (design ① + ③).

  logits = classical_main(image) + tanh(gate) · branch(image)      [gate init 0]

- classical_main : Linear(256→3) on the 16x16 image (the safe main path).
- branch (mode):
    'quantum' : C4-symmetrized quantum feature — amplitude-embed the image AND
                its 4 rotations, run the SAME PQC, average ⟨Z⟩ → C4-INVARIANT
                quantum features (approximate equivariance via group averaging).
    'sham'    : identical group averaging but a classical Linear stands in for
                the circuit (matched dim) → isolates the quantum circuit.
    'none'    : no residual (classical-only control).

Group averaging gives a rotation-invariant prior the bare classical Linear
lacks; quantum-vs-sham then tests whether the QUANTUM circuit adds anything on
top of that prior, in the few-shot regime where capacity isn't saturated.
"""
import os, sys
if "jax" not in sys.modules:
    sys.modules["jax"] = None
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

IMG = 16
N_Q = 8
K_LATENT = int(os.environ.get("QEQ_K", "7"))
N_LAYERS = int(os.environ.get("QEQ_LAYERS", "3"))
N_GROUP = int(os.environ.get("QEQ_GROUP", "4"))     # C4 rotations (set 1 to disable)

_DEV = qml.device("default.qubit", wires=N_Q)


def enc_shape():
    return qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_Q)


@qml.qnode(_DEV, interface="torch", diff_method="backprop")
def _latent_z(img_vec, weights):
    qml.AmplitudeEmbedding(img_vec, wires=range(N_Q), normalize=True, pad_with=0.0)
    qml.StronglyEntanglingLayers(weights, wires=range(N_Q))
    return [qml.expval(qml.PauliZ(i)) for i in range(K_LATENT)]


class EquivResidualHead(nn.Module):
    def __init__(self, n_classes=3, mode="quantum"):
        super().__init__()
        assert mode in ("quantum", "sham", "none")
        self.mode = mode
        self.classical = nn.Linear(IMG * IMG, n_classes)          # safe main path
        if mode == "quantum":
            self.weights = nn.Parameter(0.1 * torch.randn(enc_shape()))
        elif mode == "sham":
            self.sham = nn.Linear(IMG * IMG, K_LATENT)
        if mode != "none":
            self.q_out = nn.Linear(K_LATENT, n_classes)
            self.gate = nn.Parameter(torch.zeros(n_classes))      # zero-init → starts at classical

    def _branch_feat(self, imgs):                                  # C4-symmetrized (B,K)
        B = imgs.shape[0]
        acc = 0.0
        for k in range(N_GROUP):
            xr = torch.rot90(imgs, k, dims=(1, 2)).reshape(B, -1)
            xr = xr / (xr.norm(dim=1, keepdim=True) + 1e-9)
            if self.mode == "quantum":
                acc = acc + torch.stack(_latent_z(xr, self.weights), dim=-1).to(xr.dtype)
            else:
                acc = acc + torch.tanh(self.sham(xr))
        return acc / N_GROUP                                       # C4-invariant

    def forward(self, imgs):
        v = imgs.reshape(imgs.shape[0], -1)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-9)
        logits_c = self.classical(v)
        if self.mode == "none":
            return logits_c
        logits_q = self.q_out(self._branch_feat(imgs))
        return logits_c + torch.tanh(self.gate) * logits_q
