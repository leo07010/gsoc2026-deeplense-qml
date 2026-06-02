#!/usr/bin/env python
"""
Quantum Masked Autoencoder (QMAE) for DeepLense — prototype.

Faithful to Andrews et al. (arXiv:2511.17372), adapted to strong-lensing images:
  16x16 image → amplitude-embed into N_Q=8 qubits → encoder U(θ) → push latent
  into a fresh register via SWAP (trash qubits left as |0⟩) → decoder U†(θ) →
  reconstruct → fidelity to the ORIGINAL (unmasked) image.

Masking: a learnable mask token replaces masked 4x4 patches in classical space
before amplitude embedding (avoids mid-circuit measurement), exactly as in the
QMAE paper. Self-supervised loss = 1 − |⟨original|ρ_recon|original⟩|².

Encoder ansatz: StronglyEntanglingLayers (a clean stand-in for the paper's
Wang et al. two-qubit ansatz); decoder = its adjoint with shared weights.
"""
import os, sys
if "jax" not in sys.modules:
    sys.modules["jax"] = None
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

IMG = 16
PATCH = 4
GRID = IMG // PATCH               # 4 → 16 patches
N_Q = 8                           # 2^8 = 256 = 16*16 amplitudes
K_LATENT = int(os.environ.get("QMAE_K", "7"))   # latent qubits; trash = N_Q - K
N_LAYERS = int(os.environ.get("QMAE_LAYERS", "3"))

_DEV = qml.device("default.qubit", wires=2 * N_Q)
_BACKEND = f"QMAE default.qubit (N_Q={N_Q}, latent={K_LATENT}, layers={N_LAYERS})"


def enc_shape():
    return qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_Q)


@qml.qnode(_DEV, interface="torch", diff_method="backprop")
def _recon(masked_vec, weights):
    qml.AmplitudeEmbedding(masked_vec, wires=range(N_Q), normalize=True, pad_with=0.0)
    qml.StronglyEntanglingLayers(weights, wires=range(N_Q))            # encoder U(θ)
    for i in range(K_LATENT):                                          # move latent → fresh reg
        qml.SWAP(wires=[i, N_Q + i])                                   # trash positions stay |0⟩
    qml.adjoint(qml.StronglyEntanglingLayers)(weights, wires=range(N_Q, 2 * N_Q))  # decoder U†
    return qml.density_matrix(wires=range(N_Q, 2 * N_Q))               # (.., 256, 256)


class QuantumMAE(nn.Module):
    def __init__(self, mask_patches=(5, 6, 9, 10)):     # central 2x2 block = 25%
        super().__init__()
        self.weights = nn.Parameter(0.1 * torch.randn(enc_shape()))
        self.mask_token = nn.Parameter(torch.zeros(PATCH * PATCH))
        self.mask_patches = tuple(mask_patches)

    def apply_mask(self, imgs):                          # imgs (B,16,16)
        x = imgs.clone()
        tok = self.mask_token.view(PATCH, PATCH)
        for p in self.mask_patches:
            r = (p // GRID) * PATCH
            c = (p % GRID) * PATCH
            x[:, r:r + PATCH, c:c + PATCH] = tok
        return x

    def reconstruct_fidelity(self, imgs):                # imgs (B,16,16) → fidelity (B,)
        B = imgs.shape[0]
        masked = self.apply_mask(imgs).reshape(B, -1)            # (B,256)
        orig = imgs.reshape(B, -1)
        orig = orig / (orig.norm(dim=1, keepdim=True) + 1e-9)    # normalized target
        rho = _recon(masked, self.weights)                       # (B,256,256) complex
        orig_c = orig.to(rho.dtype)
        tmp = torch.einsum("bij,bj->bi", rho, orig_c)
        fid = torch.einsum("bi,bi->b", orig_c.conj(), tmp).real  # ⟨o|ρ|o⟩
        return fid

    def forward(self, imgs):
        return 1.0 - self.reconstruct_fidelity(imgs)             # loss per sample


# ══════════════════════════════════════════════════════════════════════
# Downstream 3-class: latent ⟨Z⟩ → classical head, with a sham control
# ══════════════════════════════════════════════════════════════════════
_DEV_C = qml.device("default.qubit", wires=N_Q)


@qml.qnode(_DEV_C, interface="torch", diff_method="backprop")
def _latent_z(img_vec, weights):
    """Amplitude-embed full image → encoder U(θ) → ⟨Z_i⟩ of the K_LATENT latent qubits."""
    qml.AmplitudeEmbedding(img_vec, wires=range(N_Q), normalize=True, pad_with=0.0)
    qml.StronglyEntanglingLayers(weights, wires=range(N_Q))
    return [qml.expval(qml.PauliZ(i)) for i in range(K_LATENT)]


class QMAEClassifier(nn.Module):
    """Full 16x16 image → K_LATENT-dim latent → classical head → 3 classes.

    quantum (sham=False): latent = ⟨Z⟩ of the quantum encoder's latent qubits.
    sham    (sham=True) : latent = classical Linear(256→K_LATENT)+tanh (matched
                          bottleneck, NO circuit) — isolates quantum vs capacity.
    """

    def __init__(self, n_classes=3, sham=False):
        super().__init__()
        self.sham = sham
        if sham:
            self.enc = nn.Linear(IMG * IMG, K_LATENT)
        else:
            self.weights = nn.Parameter(0.1 * torch.randn(enc_shape()))
        self.head = nn.Sequential(nn.LayerNorm(K_LATENT), nn.Linear(K_LATENT, n_classes))

    def latent(self, imgs):
        B = imgs.shape[0]
        v = imgs.reshape(B, -1)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-9)          # match amplitude-embed norm
        if self.sham:
            return torch.tanh(self.enc(v))                    # (B, K) classical, range [-1,1]
        z = torch.stack(_latent_z(v, self.weights), dim=-1)   # (B, K) ⟨Z⟩ in [-1,1]
        return z.to(v.dtype)

    def forward(self, imgs):
        return self.head(self.latent(imgs))


class ClassicalAE(nn.Module):
    """Sham control for quantum-AE anomaly detection: classical Linear encoder/decoder
    with the SAME latent dim K and the SAME normalized-overlap² fidelity metric."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(IMG * IMG, K_LATENT)
        self.dec = nn.Linear(K_LATENT, IMG * IMG)

    def reconstruct_fidelity(self, imgs):
        B = imgs.shape[0]
        x = imgs.reshape(B, -1)
        xn = x / (x.norm(dim=1, keepdim=True) + 1e-9)
        z = torch.tanh(self.enc(xn))
        r = self.dec(z)
        rn = r / (r.norm(dim=1, keepdim=True) + 1e-9)
        return (rn * xn).sum(1) ** 2                      # overlap² ∈ [0,1]

    def forward(self, imgs):
        return 1.0 - self.reconstruct_fidelity(imgs)
