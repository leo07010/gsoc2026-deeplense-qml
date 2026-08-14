#!/usr/bin/env python3
"""Quantum Patch Embedding (QPE) experiment -- 2026-08-14.

Motivated by experiments/09_spectral_diagnostic: the class-discriminative
signal lives in INTRA-patch spatial frequencies (radial bins 7-29, i.e.
2-9 px wavelengths; Cohen's d up to 2.86), and is DESTROYED by pooling to
the 8x8 token grid (token-axis d < 0.4 everywhere). So the quantum
circuit is placed at the patch-EMBEDDING stage, acting on the raw 64-pixel
vector of each 8x8 patch, where its bit-stride structure has physical
meaning:

  pixel (i, j) within a patch -> basis index b = i*8 + j
  bits 5..3 of b = i (row), bits 2..0 = j (col)
  RYCNOTLayer convention: qubit q acts on bit (5 - q), so
    qubit 0 mixes top/bottom patch halves   (row stride 4)
    qubit 1 mixes row stride 2
    qubit 2 mixes adjacent rows             (row stride 1)
    qubit 3 mixes left/right patch halves   (col stride 4)
    qubit 4 mixes col stride 2
    qubit 5 mixes adjacent cols             (col stride 1)
  and the CNOT ring couples the scales (incl. q5->q0: finest col scale
  to coarsest row scale).

Four embedding arms; EVERYTHING downstream is identical (CLS token +
learned pos emb + DEPTH standard transformer blocks with classical MHA +
linear head), so the arms differ ONLY in how a patch's 64 pixels become a
64-dim token:

  quantum  : token = U(theta) @ pixels, U from the 6-qubit RY+CNOT ring
             circuit (48 trainable angles, shared across patches).
  scramble : same circuit, but the pixel->basis-state assignment is a
             fixed random permutation (drawn from the training seed).
             Same param count, same orthogonality -- only the bit-stride
             spatial alignment is destroyed. THE falsification control:
             if quantum > scramble, the circuit's structure (not just
             orthogonality/low params) is doing real work.
  dct      : fixed orthogonal 2D DCT-II (kron(DCT8, DCT8)), 0 params.
             The classical spectral analog of what the circuit can learn.
  conv     : learned Conv2d(1, 64, 8, 8) -- the standard ViT patch embed
             used by every previous experiment in this repo (4,160 params).

DESIGN CONSTRAINT (do not "fix"): there is deliberately NO learned linear
layer between the embedding and the first attention block for the
quantum/scramble/dct arms -- a learned Linear(64,64) would absorb any
orthogonal transform (Linear o U spans the same function class as Linear
alone) and erase the very difference this experiment measures. The
additive learned positional embedding provides a per-token bias without
absorbing rotations. The three non-conv arms DO get one scalar learnable
gain (1 param): review found conv's unconstrained weight norm lets it
tune the token:positional-embedding ratio while norm-preserving arms
cannot -- a scalar gain closes that confound and cannot absorb an
orthogonal transform.

Training scaffolding (split_val_into_sel_and_test, evaluate, best-val_sel
checkpoint, test touched exactly once) copied verbatim from
experiments/08_qovt_ablation/train_qovt_ablation.py so the evaluation
protocol is identical to the pre-registered QOVT ablation.
"""
import os, math, argparse, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize

IMG        = 64
PATCH_SIZE = 8
D          = 64            # = PATCH_SIZE**2: one channel per intra-patch pixel/mode
DEPTH      = int(os.environ.get("QPE_DEPTH", "4"))
MLP_MULT   = 4
N_HEADS    = 8


# ── RY+CNOT circuit: verbatim copy of RYCNOTLayer from
# experiments/08_qovt_ablation/train_qovt_ablation.py (same math, same
# qubit/bit convention), so this experiment tests the SAME circuit the
# pre-registered ablation reported on, applied at a different stage. ──
class RYCNOTLayer(nn.Module):
    """6-qubit RY+CNOT ring circuit expressed as a 64x64 PyTorch matrix.
    48 trainable angles (8 layers x 6 qubits)."""
    N_Q = 6

    def __init__(self, D: int, n_layers: int = 8):
        super().__init__()
        assert D == 64, f"RYCNOTLayer requires D=64 (2^6), got {D}"
        self.D = D
        self.N_LAYERS = n_layers
        self.angles = nn.Parameter(0.01 * torch.randn(self.N_LAYERS, self.N_Q))
        self.register_buffer('_cnot_ring', self._build_cnot_ring())

    def _build_cnot_ring(self) -> torch.Tensor:
        D = self.D
        W = torch.eye(D)
        for ctrl in range(self.N_Q):
            tgt      = (ctrl + 1) % self.N_Q
            ctrl_bit = self.N_Q - 1 - ctrl
            tgt_bit  = self.N_Q - 1 - tgt
            perm = torch.zeros(D, D)
            for s in range(D):
                s2 = s ^ (1 << tgt_bit) if (s >> ctrl_bit) & 1 else s
                perm[s2, s] = 1.0
            W = perm @ W
        return W

    def get_matrix(self) -> torch.Tensor:
        D    = self.D
        W    = torch.eye(D, dtype=self.angles.dtype, device=self.angles.device)
        cnot = self._cnot_ring.to(dtype=self.angles.dtype)
        ca   = torch.cos(self.angles / 2)
        sa   = torch.sin(self.angles / 2)
        for layer in range(self.N_LAYERS):
            R = torch.zeros(self.N_Q, 2, 2, dtype=self.angles.dtype,
                            device=self.angles.device)
            R[:, 0, 0] =  ca[layer];  R[:, 0, 1] = -sa[layer]
            R[:, 1, 0] =  sa[layer];  R[:, 1, 1] =  ca[layer]
            Wr = W.view(*([2] * self.N_Q), D)
            for q in range(self.N_Q):
                Wr = torch.tensordot(R[q], Wr, dims=([1], [q])).movedim(0, q)
            W = Wr.contiguous().view(D, D)
            W = cnot @ W
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.get_matrix().T


def dct2_matrix(n: int = PATCH_SIZE) -> torch.Tensor:
    """Orthonormal 1D DCT-II matrix (n x n); 2D version is kron(D1, D1)."""
    k = torch.arange(n).float()
    M = torch.cos(math.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    M[0]  *= math.sqrt(1.0 / n)
    M[1:] *= math.sqrt(2.0 / n)
    return M


def extract_patches(x: torch.Tensor) -> torch.Tensor:
    """(B, 1, 64, 64) or (B, 64, 64) -> (B, 64 tokens, 64 pixels).
    Token t = r*8 + c (patch grid, row-major); pixel p = i*8 + j
    (within-patch, row-major) -- the bit mapping documented above."""
    if x.dim() == 4:
        x = x[:, 0]
    B = x.shape[0]
    x = x.view(B, 8, PATCH_SIZE, 8, PATCH_SIZE)   # (B, r, i, c, j)
    x = x.permute(0, 1, 3, 2, 4).contiguous()     # (B, r, c, i, j)
    return x.view(B, 64, D)


class PatchEmbed(nn.Module):
    def __init__(self, mode: str, seed: int, n_circuit_layers: int = 8):
        super().__init__()
        self.mode = mode
        if mode == "quantum":
            self.circ = RYCNOTLayer(D, n_circuit_layers)
        elif mode == "scramble":
            self.circ = RYCNOTLayer(D, n_circuit_layers)
            perm = torch.from_numpy(
                np.random.default_rng(9000 + seed).permutation(D))
            self.register_buffer("perm", perm)
        elif mode == "dct":
            M1 = dct2_matrix()
            self.register_buffer("dct", torch.kron(M1, M1))
        elif mode == "conv":
            self.conv = nn.Conv2d(1, D, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        else:
            raise ValueError(f"Unknown embed mode: {mode}")
        if mode != "conv":
            # scalar gain: lets norm-preserving arms tune token scale the
            # way conv's unconstrained weights can; cannot absorb a rotation
            self.gain = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "conv":
            if x.dim() == 3:
                x = x.unsqueeze(1)
            return self.conv(x).flatten(2).transpose(1, 2)   # (B, 64, D)
        p = extract_patches(x)                               # (B, 64, 64)
        if self.mode == "quantum":
            return self.gain * self.circ(p)
        if self.mode == "scramble":
            return self.gain * self.circ(p[:, :, self.perm])
        return self.gain * (p @ self.dct.T)                  # dct


class MLP(nn.Module):
    def __init__(self, Dm: int, mult: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(Dm), nn.Linear(Dm, Dm * mult), nn.GELU(),
            nn.Linear(Dm * mult, Dm))

    def forward(self, x):
        return self.net(x)


class MHABlockAttn(nn.Module):
    def __init__(self, Dm: int, n_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(Dm)
        self.mha  = nn.MultiheadAttention(Dm, num_heads=n_heads,
                                          batch_first=True, bias=False)

    def forward(self, x):
        xn = self.norm(x)
        out, _ = self.mha(xn, xn, xn)
        return out


class QPEViT(nn.Module):
    """Identical classical ViT backbone for all four arms; only the patch
    embedding differs."""

    def __init__(self, n_classes: int, embed: str, seed: int,
                 n_circuit_layers: int = 8):
        super().__init__()
        N_patches  = (IMG // PATCH_SIZE) ** 2
        # backbone FIRST, embed LAST: different embed modes consume
        # different amounts of the torch RNG stream, so building the embed
        # first would give each arm a different backbone init -- building
        # it last makes the backbone init bit-identical across all 4 arms
        # for the same seed (reduces unpaired variance in the arm tests)
        self.cls   = nn.Parameter(torch.zeros(1, 1, D))
        self.pos   = nn.Parameter(torch.randn(1, N_patches + 1, D) * 0.02)
        self.attn_layers = nn.ModuleList(
            [MHABlockAttn(D, N_HEADS) for _ in range(DEPTH)])
        self.mlp_layers  = nn.ModuleList([MLP(D, MLP_MULT) for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, n_classes)
        self.embed = PatchEmbed(embed, seed, n_circuit_layers)

    def embed_parameters(self):
        return list(self.embed.parameters())

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self.embed(imgs)
        B = x.shape[0]
        cls = self.cls.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos
        for attn, mlp in zip(self.attn_layers, self.mlp_layers):
            x = x + attn(x)
            x = x + mlp(x)
        return self.head(self.norm(x[:, 0]))


# ── evaluation + split: verbatim from train_qovt_ablation.py ──
def evaluate(model, x, y, device, C, bs=64):
    model.eval(); logits = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            logits.append(model(x[i:i+bs].to(device)).cpu())
    probs = torch.softmax(torch.cat(logits), 1).numpy()
    yy    = y.numpy()
    yb    = label_binarize(yy, classes=np.arange(C))
    return dict(
        auc=roc_auc_score(yb, probs, average="macro", multi_class="ovr"),
        auc_per=roc_auc_score(yb, probs, average=None, multi_class="ovr"),
        acc=accuracy_score(yy, probs.argmax(1)),
        f1=f1_score(yy, probs.argmax(1), average="macro"),
        cm=confusion_matrix(yy, probs.argmax(1)),
    )


def split_val_into_sel_and_test(vx, vy, split_seed: int):
    n = len(vy)
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    half = n // 2
    sel_idx, test_idx = perm[:half], perm[half:]
    return (vx[sel_idx], vy[sel_idx]), (vx[test_idx], vy[test_idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",        required=True)
    ap.add_argument("--embed",       required=True,
                    choices=["quantum", "scramble", "dct", "conv"])
    ap.add_argument("--epochs",      type=int,   default=48)
    ap.add_argument("--batch_size",  type=int,   default=64)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--split_seed",  type=int,   default=0)
    ap.add_argument("--n_per_class", type=int,   default=0)
    ap.add_argument("--circuit_layers", type=int, default=8)
    ap.add_argument("--smoke",       action="store_true")
    ap.add_argument("--out_json",    default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d  = np.load(args.data, allow_pickle=True)
    cn = list(d["class_names"]); C = len(cn)
    tx = torch.from_numpy(d["train_x"]).float()
    ty = torch.from_numpy(d["train_y"]).long()
    vx_all = torch.from_numpy(d["val_x"]).float()
    vy_all = torch.from_numpy(d["val_y"]).long()
    (vx, vy), (tex, tey) = split_val_into_sel_and_test(vx_all, vy_all, args.split_seed)

    if args.n_per_class > 0:
        rng = np.random.default_rng(500 + args.seed)
        tn  = ty.numpy()
        idx = np.concatenate([
            rng.choice(np.where(tn == c)[0],
                       min(args.n_per_class, (tn == c).sum()), replace=False)
            for c in range(C)
        ])
        tx, ty = tx[idx], ty[idx]
    if args.smoke:
        tx, ty = tx[:128], ty[:128]; vx, vy = vx[:256], vy[:256]
        tex, tey = tex[:256], tey[:256]; args.epochs = 2

    model = QPEViT(C, embed=args.embed, seed=args.seed,
                   n_circuit_layers=args.circuit_layers).to(device)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_embed = sum(p.numel() for p in model.embed_parameters() if p.requires_grad)

    # snapshot circuit init state so a null result can be diagnosed as
    # "structure doesn't help" vs "the angles never moved" (near-zero init
    # makes U(theta_0) ~ the CNOT-ring permutation, which the
    # permutation-equivariant backbone can absorb)
    init_angles, U_init = None, None
    if args.embed in ("quantum", "scramble"):
        with torch.no_grad():
            init_angles = model.embed.circ.angles.detach().cpu().clone()
            U_init      = model.embed.circ.get_matrix().detach().cpu().clone()

    # single uniform lr for every parameter, every arm (lesson from the
    # 08 ablation's invalidated run 1: no per-arm LR special-casing)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(10, args.epochs // 3), T_mult=2, eta_min=1e-6)
    crit  = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(tx, ty),
                        batch_size=min(args.batch_size, len(tx)), shuffle=True)

    base = os.path.splitext(os.path.basename(args.data))[0]
    print(f"[INFO] QPE D={D} P={PATCH_SIZE} depth={DEPTH} | embed={args.embed} "
          f"circuit_layers={args.circuit_layers if args.embed in ('quantum','scramble') else 'n/a'} | "
          f"data={base} N/class={args.n_per_class or 'full'} seed={args.seed} "
          f"split_seed={args.split_seed} | params={n_total} embed_params={n_embed} "
          f"lr={args.lr} epochs={args.epochs} | train{tuple(tx.shape)} "
          f"val_sel{tuple(vx.shape)} test{tuple(tex.shape)} device={device}", flush=True)

    best_sel_auc = -1.0
    best_state   = None
    for ep in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        sched.step()
        m = evaluate(model, vx, vy, device, C)          # val_sel ONLY -- selection
        if m["auc"] > best_sel_auc:
            best_sel_auc = m["auc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"[ep {ep+1:03d}] val_sel AUC={m['auc']:.4f} acc={m['acc']:.4f}", flush=True)

    model.load_state_dict(best_state)
    test_m = evaluate(model, tex, tey, device, C)

    angle_drift, U_drift = None, None
    if init_angles is not None:
        with torch.no_grad():
            angle_drift = (model.embed.circ.angles.detach().cpu()
                           - init_angles).norm().item()
            U_drift = (model.embed.circ.get_matrix().detach().cpu()
                       - U_init).norm().item()

    per_s = " ".join(f"{cn[i]}={test_m['auc_per'][i]:.4f}" for i in range(len(cn)))
    print(f"{'='*60}\nTEST (best val_sel epoch, held-out, evaluated once)\n{'='*60}")
    print(f"[DONE-QPE] embed={args.embed} {base} N={args.n_per_class or 'full'} "
          f"seed={args.seed} test_AUC={test_m['auc']:.4f} val_sel_AUC={best_sel_auc:.4f} "
          f"per_class=({per_s}) params={n_total} embed_params={n_embed} "
          f"angle_drift={angle_drift} U_frob_drift={U_drift}", flush=True)

    if args.out_json:
        with open(args.out_json, "a") as f:
            f.write(json.dumps(dict(
                embed=args.embed, data=base,
                n_per_class=args.n_per_class or "full",
                seed=args.seed, split_seed=args.split_seed,
                test_auc=test_m["auc"], val_sel_auc=best_sel_auc,
                test_auc_per=test_m["auc_per"].tolist(), params=n_total,
                embed_params=n_embed,
                depth=DEPTH, epochs=args.epochs, lr=args.lr,
                circuit_layers=args.circuit_layers,
                angle_drift=angle_drift, U_frob_drift=U_drift)) + "\n")


if __name__ == "__main__":
    main()
