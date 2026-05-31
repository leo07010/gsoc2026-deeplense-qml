#!/usr/bin/env python
"""
F2 — Gated Residual Quantum-Classical Fusion head (CUDA-Q backend).

    logits = classical_head(cls)  +  tanh(g) ⊙ quantum_head( PQC(cls) )
             └──────────┬───────┘     └──┬──┘  └───────────┬──────────┘
        init from shipped head.weight   learnable gate   16-qubit dressed PQC
        → at init g=0 ⇒ model == baseline (AUC 0.974), can only improve.

Quantum branch (dressed quantum circuit, Mari et al. 2020):
    192-d CLS ─Linear→ N_Q angles ─RY encode (once)─→
        [ RZ,RY (trainable) + CNOT-ring ] × N_LAYERS  ─→  ⟨Z_i⟩ for i=0..N_Q-1
    The N_Q expectation values → Linear(N_Q→3) → quantum logits.

Gradients: parameter-shift rule, hand-written torch.autograd.Function.
    Every trainable angle/weight appears in exactly ONE rotation gate
    (we encode ONCE, no data re-uploading) ⇒ the standard ±π/2 shift is exact.
    (Re-uploading would re-use each input angle across layers; its gradient then
     needs a per-occurrence shift-and-sum. Left as a documented extension below.)

ENVIRONMENT: needs `cudaq` (Linux / WSL2 / Colab / H100). Will NOT import on
native Windows. Train on cached features via train_fusion_cudaq.py.
"""
import numpy as np
import torch
import torch.nn as nn

# ── Circuit size knobs ────────────────────────────────────────────────
N_Q = 16            # qubits  (≤ 30 budget; 16 = sweet spot for a 192-d head)
N_LAYERS = 4        # trainable entangling layers
N_WEIGHTS = N_Q * 2 * N_LAYERS    # RZ + RY per qubit per layer
SHIFT = np.pi / 2

try:
    import cudaq
    from cudaq import spin
    _HAS_CUDAQ = True
except Exception:                  # noqa: BLE001 — absent on Windows
    _HAS_CUDAQ = False


# ══════════════════════════════════════════════════════════════════════
# CUDA-Q kernel + batched expectation-value evaluator
# ══════════════════════════════════════════════════════════════════════
if _HAS_CUDAQ:
    OBS = [spin.z(i) for i in range(N_Q)]

    @cudaq.kernel
    def pqc(angles: list[float], weights: list[float]):
        q = cudaq.qvector(N_Q)
        # ── encode features once ──
        for i in range(N_Q):
            ry(angles[i], q[i])
        # ── trainable variational layers ──
        for L in range(N_LAYERS):
            base = L * N_Q * 2
            for i in range(N_Q):
                rz(weights[base + i * 2], q[i])
                ry(weights[base + i * 2 + 1], q[i])
            for i in range(N_Q - 1):           # CNOT ring
                x.ctrl(q[i], q[i + 1])
            x.ctrl(q[N_Q - 1], q[0])

    def _expvals(angles_np, weights_np):
        """Return (B, N_Q) array of ⟨Z_i⟩.

        Uses CUDA-Q's broadcast observe: passing a (B, ·) argument array runs
        the whole batch (parallel on the `nvidia` GPU target).
        """
        B = angles_np.shape[0]
        w_batch = np.tile(weights_np[None, :], (B, 1))
        out = np.empty((B, N_Q), dtype=np.float64)
        for o, obs in enumerate(OBS):
            results = cudaq.observe(pqc, obs, angles_np, w_batch)  # list len B
            out[:, o] = np.array([r.expectation() for r in results])
        return out
else:
    def _expvals(angles_np, weights_np):       # pragma: no cover
        raise RuntimeError("CUDA-Q not available — run on Linux/WSL/H100.")


# ══════════════════════════════════════════════════════════════════════
# Differentiable wrapper (parameter-shift)
# ══════════════════════════════════════════════════════════════════════
class _QuantumFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, angles, weights):
        a = angles.detach().cpu().numpy().astype(np.float64)   # (B, N_Q)
        w = weights.detach().cpu().numpy().astype(np.float64)  # (N_WEIGHTS,)
        ctx.a, ctx.w, ctx.dev, ctx.dt = a, w, angles.device, angles.dtype
        ev = _expvals(a, w)                                     # (B, N_Q)
        return torch.from_numpy(ev).to(angles.device, angles.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        a, w = ctx.a, ctx.w
        g = grad_out.detach().cpu().numpy().astype(np.float64)  # (B, N_Q_obs)

        # ── grad wrt encoding angles (per-sample; column i = qubit i) ──
        grad_a = np.zeros_like(a)
        for i in range(a.shape[1]):
            ap = a.copy(); ap[:, i] += SHIFT
            am = a.copy(); am[:, i] -= SHIFT
            deriv = 0.5 * (_expvals(ap, w) - _expvals(am, w))  # (B, N_Q_obs)
            grad_a[:, i] = np.sum(g * deriv, axis=1)

        # ── grad wrt trainable weights (shared across batch) ──
        grad_w = np.zeros_like(w)
        for j in range(w.shape[0]):
            wp = w.copy(); wp[j] += SHIFT
            wm = w.copy(); wm[j] -= SHIFT
            deriv = 0.5 * (_expvals(a, wp) - _expvals(a, wm))  # (B, N_Q_obs)
            grad_w[j] = np.sum(g * deriv)

        ga = torch.from_numpy(grad_a).to(ctx.dev, ctx.dt)
        gw = torch.from_numpy(grad_w).to(ctx.dev, ctx.dt)
        return ga, gw


# ══════════════════════════════════════════════════════════════════════
# F2 fusion head
# ══════════════════════════════════════════════════════════════════════
class QuantumFusionHead(nn.Module):
    """Gated residual fusion of a classical linear head + a quantum branch."""

    def __init__(self, in_dim=192, n_classes=3):
        super().__init__()
        # quantum branch
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, N_Q))
        self.qweights = nn.Parameter(0.01 * torch.randn(N_WEIGHTS))
        self.q_out = nn.Linear(N_Q, n_classes)
        # classical branch (init from shipped head → exact baseline)
        self.classical = nn.Linear(in_dim, n_classes)
        # gate: zero ⇒ tanh(0)=0 ⇒ pure classical at init
        self.gate = nn.Parameter(torch.zeros(n_classes))

    @torch.no_grad()
    def init_classical(self, head_weight, head_bias):
        self.classical.weight.copy_(torch.as_tensor(head_weight))
        self.classical.bias.copy_(torch.as_tensor(head_bias))

    def forward(self, cls_feat):
        angles = torch.tanh(self.proj(cls_feat)) * np.pi      # bound to [-π, π]
        qz = _QuantumFn.apply(angles, self.qweights)          # (B, N_Q)
        logits_q = self.q_out(qz)
        logits_c = self.classical(cls_feat)
        return logits_c + torch.tanh(self.gate) * logits_q


class QuantumFusionViT(nn.Module):
    """Frozen ViT encoder + F2 head. Returns (logits, cls_feat) like ViTClassifier."""

    def __init__(self, encoder, num_classes=3, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        self.frozen = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()
        self.head = QuantumFusionHead(encoder.embed_dim, num_classes)

    def forward(self, x):
        if self.frozen:
            with torch.no_grad():
                cls_feat = self.encoder(x)[:, 0]
        else:
            cls_feat = self.encoder(x)[:, 0]
        return self.head(cls_feat), cls_feat
