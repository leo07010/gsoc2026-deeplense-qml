#!/usr/bin/env python3
"""QPE-2: spectrally-initialized quantum patch embedding -- 2026-08-19.

QPE-1 (experiments/10_quantum_patch_embed) established that (a) a fixed 2D
DCT patch embedding is the strongest arm at N=500 -- the intra-patch
spectral prior from the Step-0 diagnostic is the active ingredient -- and
(b) a randomly-initialized RY+CNOT circuit cannot beat it (it starts at a
spectrally useless point, the CNOT-ring permutation). QPE-2 asks the
remaining winnable question: at EXTREME low N (100-250/class), does a
learnable orthogonal refinement AROUND the DCT prior beat the fixed prior
-- and if so, is the quantum circuit chart the best way to parameterize
that refinement?

Every refinement arm is the composition  U = R(params) @ T_dct  with the
refinement initialized at (a relabeling of) the identity, so all arms
start from the same spectral prior and differ ONLY in the chart used for
the 2016-dim manifold O(64) around it:

  quantum   : R = RY+CNOT circuit (48 angles, init EXACTLY zero -> R0 is
              the CNOT-ring permutation P^8, a pure channel relabeling
              of the DCT; the backbone init distribution is exchangeable
              under channel permutations, so this arm's starting point
              is distributionally identical to dctfix). A genuine
              gate-model circuit; the fixed DCT block is also
              circuit-implementable (quantum DCT via QFT, Klappenecker
              & Roetteler 2001), so the composition remains a
              legitimate quantum circuit. Gradients at theta=0 are
              verified nonzero for all 48 angles.
  skew48    : R = Cayley chart restricted to 48 fixed skew entries
              (deterministic rng, same 48 pairs for every run), init 0
              -> R0 = I exactly -- the PARAM-MATCHED classical chart
              (48 vs quantum's 48; added on adversarial review so a
              quantum-vs-cayley win cannot be explained by "fewer
              params generalize better").
  butterfly : R = Givens butterfly (192 angles, init 0 -> R0 = I exactly)
              -- the classical orthogonal chart this project already
              showed is unstable in the QOVT setting (verbatim copy of
              08's ButterflyLayer).
  cayley    : R = (I-A)(I+A)^{-1}, A skew from 2016 free params, init 0
              -> R0 = I exactly -- the full-dimensional classical chart
              (strong classical: 42x more params than quantum).
  dctfix    : R = I frozen -- the QPE-1 winner, the bar to beat.
  conv      : learned Conv2d patch embed (no prior) -- context baseline.

Param ladder (embed, incl. the scalar gain each non-conv arm carries,
same rationale as QPE-1): quantum 49 = skew48 49 < butterfly 193 <
cayley 2017 < conv 4160; dctfix 1. Backbone (CLS + pos + 4
classical-MHA blocks + head) is byte-identical across arms and built
BEFORE the embed so its init is bit-identical per seed (QPE-1 lesson).

Training scaffolding (held-out split, best-val_sel checkpoint, test
touched once) identical to QPE-1 / the QOVT ablation. Epochs raised to
96 (ends exactly at a cosine-restart cycle end: T_0=32, T_mult=2) and
batch size lowered to 32 because N=100/class is only 300 images.
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
D          = 64
DEPTH      = int(os.environ.get("QPE2_DEPTH", "4"))
MLP_MULT   = 4
N_HEADS    = 8


# ── RY+CNOT circuit: verbatim copy of QPE-1's RYCNOTLayer (itself a
# verbatim copy of 08's) -- same math, same qubit/bit convention. ──
class RYCNOTLayer(nn.Module):
    N_Q = 6

    def __init__(self, D: int, n_layers: int = 8):
        super().__init__()
        assert D == 64, f"RYCNOTLayer requires D=64 (2^6), got {D}"
        self.D = D
        self.N_LAYERS = n_layers
        # init EXACTLY zero (QPE-1 used 0.01*randn): at theta=0 the circuit
        # is the pure CNOT-ring permutation, so this arm starts at a channel
        # relabeling of the DCT while the classical charts start at exactly
        # I -- making "the arms differ only in the chart" literally true.
        # Gradients at theta=0 are nonzero for all 48 angles (unit-tested).
        self.angles = nn.Parameter(torch.zeros(self.N_LAYERS, self.N_Q))
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


# ── Givens butterfly: verbatim copy of 08's ButterflyLayer. ──
class ButterflyLayer(nn.Module):
    def __init__(self, D: int):
        super().__init__()
        assert D > 0 and (D & (D - 1)) == 0, f"D={D} must be a power of 2"
        self.D     = D
        self.log2D = int(math.log2(D))
        n_gates    = (D // 2) * self.log2D
        self.angles = nn.Parameter(torch.zeros(n_gates))
        for stage in range(self.log2D):
            step = 2 ** (stage + 1)
            half = step // 2
            i_list, j_list = [], []
            for start in range(0, D, step):
                for k in range(half):
                    i_list.append(start + k)
                    j_list.append(start + k + half)
            self.register_buffer(f'_si_{stage}', torch.tensor(i_list, dtype=torch.long))
            self.register_buffer(f'_sj_{stage}', torch.tensor(j_list, dtype=torch.long))

    def get_matrix(self) -> torch.Tensor:
        W   = torch.eye(self.D, dtype=self.angles.dtype, device=self.angles.device)
        idx = 0
        for stage in range(self.log2D):
            i_t = getattr(self, f'_si_{stage}')
            j_t = getattr(self, f'_sj_{stage}')
            n   = i_t.shape[0]
            thetas = self.angles[idx: idx + n]
            c = torch.cos(thetas)
            s = torch.sin(thetas)
            Wi = W[:, i_t].clone()
            Wj = W[:, j_t].clone()
            W  = W.clone()
            W[:, i_t] = Wi * c.unsqueeze(0) - Wj * s.unsqueeze(0)
            W[:, j_t] = Wi * s.unsqueeze(0) + Wj * c.unsqueeze(0)
            idx += n
        return W


class CayleyLayer(nn.Module):
    """Orthogonal chart R = (I - A)(I + A)^{-1} with A skew-symmetric,
    init 0 -> R = I exactly. Note (I-A)(I+A)^{-1} == (I+A)^{-1}(I-A) for
    skew A, since (I+A)(I-A) = I - A^2 = (I-A)(I+A) -- so the solve()
    form below is the same matrix as the docstring form.

    n_params=None -> full chart (D*(D-1)/2 = 2016 free params).
    n_params=k    -> the param-MATCHED chart: only k of the 2016 skew
                     entries are free (chosen once by a fixed seed, so
                     every run/arm/seed uses the SAME k coordinates),
                     the rest pinned to zero. Used for skew48 (k=48),
                     added on adversarial review: without it, a
                     quantum(48) > cayley(2016) win at N=100 is more
                     parsimoniously explained by parameter count than
                     by anything about the circuit chart."""

    def __init__(self, D: int, n_params: int = None, coord_seed: int = 0):
        super().__init__()
        self.D = D
        n_full = D * (D - 1) // 2
        iu = torch.triu_indices(D, D, offset=1)
        if n_params is None or n_params >= n_full:
            self.n = n_full
            self.register_buffer('_iu', iu)
        else:
            self.n = n_params
            sel = torch.from_numpy(
                np.random.default_rng(coord_seed).choice(
                    n_full, size=n_params, replace=False))
            sel = sel.sort().values
            self.register_buffer('_iu', iu[:, sel])
        self.w = nn.Parameter(torch.zeros(self.n))

    def get_matrix(self) -> torch.Tensor:
        D = self.D
        A = torch.zeros(D, D, dtype=self.w.dtype, device=self.w.device)
        A[self._iu[0], self._iu[1]] = self.w
        A = A - A.T
        I = torch.eye(D, dtype=self.w.dtype, device=self.w.device)
        return torch.linalg.solve(I + A, I - A)


def dct2_matrix(n: int = PATCH_SIZE) -> torch.Tensor:
    k = torch.arange(n).float()
    M = torch.cos(math.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    M[0]  *= math.sqrt(1.0 / n)
    M[1:] *= math.sqrt(2.0 / n)
    return M


def extract_patches(x: torch.Tensor) -> torch.Tensor:
    """(B,1,64,64) or (B,64,64) -> (B, 64 tokens, 64 pixels);
    t = r*8+c, p = i*8+j (same mapping as QPE-1, unit-tested there)."""
    if x.dim() == 4:
        x = x[:, 0]
    B = x.shape[0]
    x = x.view(B, 8, PATCH_SIZE, 8, PATCH_SIZE)
    x = x.permute(0, 1, 3, 2, 4).contiguous()
    return x.view(B, 64, D)


class PatchEmbed2(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        if mode == "conv":
            self.refine = None      # so refine_matrix() is safe for every mode
            self.conv = nn.Conv2d(1, D, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
            return
        M1 = dct2_matrix()
        self.register_buffer("dct", torch.kron(M1, M1))
        self.gain = nn.Parameter(torch.ones(1))
        if mode == "quantum":
            self.refine = RYCNOTLayer(D)
        elif mode == "skew48":
            self.refine = CayleyLayer(D, n_params=48)
        elif mode == "butterfly":
            self.refine = ButterflyLayer(D)
        elif mode == "cayley":
            self.refine = CayleyLayer(D)
        elif mode == "dctfix":
            self.refine = None
        else:
            raise ValueError(f"Unknown embed mode: {mode}")

    def refine_matrix(self):
        return None if self.refine is None else self.refine.get_matrix()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "conv":
            if x.dim() == 3:
                x = x.unsqueeze(1)
            return self.conv(x).flatten(2).transpose(1, 2)
        p = extract_patches(x)
        y = p @ self.dct.T                       # spectral prior
        if self.refine is not None:
            y = y @ self.refine.get_matrix().T   # orthogonal refinement
        return self.gain * y


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


class QPE2ViT(nn.Module):
    def __init__(self, n_classes: int, embed: str):
        super().__init__()
        N_patches = (IMG // PATCH_SIZE) ** 2
        # backbone first, embed last: bit-identical backbone init per seed
        self.cls   = nn.Parameter(torch.zeros(1, 1, D))
        self.pos   = nn.Parameter(torch.randn(1, N_patches + 1, D) * 0.02)
        self.attn_layers = nn.ModuleList(
            [MHABlockAttn(D, N_HEADS) for _ in range(DEPTH)])
        self.mlp_layers  = nn.ModuleList([MLP(D, MLP_MULT) for _ in range(DEPTH)])
        self.norm  = nn.LayerNorm(D)
        self.head  = nn.Linear(D, n_classes)
        self.embed = PatchEmbed2(embed)

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
                    choices=["quantum", "skew48", "butterfly", "cayley",
                             "dctfix", "conv"])
    ap.add_argument("--epochs",      type=int,   default=96)
    ap.add_argument("--batch_size",  type=int,   default=32)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--split_seed",  type=int,   default=0)
    ap.add_argument("--n_per_class", type=int,   default=0)
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
        tx, ty = tx[:96], ty[:96]; vx, vy = vx[:256], vy[:256]
        tex, tey = tex[:256], tey[:256]; args.epochs = 2

    model = QPE2ViT(C, embed=args.embed).to(device)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_embed = sum(p.numel() for p in model.embed_parameters() if p.requires_grad)

    R_init = None
    if model.embed.refine is not None:
        with torch.no_grad():
            R_init = model.embed.refine_matrix().detach().cpu().clone()

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(10, args.epochs // 3), T_mult=2, eta_min=1e-6)
    crit  = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(tx, ty),
                        batch_size=min(args.batch_size, len(tx)), shuffle=True)

    base = os.path.splitext(os.path.basename(args.data))[0]
    print(f"[INFO] QPE2 D={D} P={PATCH_SIZE} depth={DEPTH} | embed={args.embed} | "
          f"data={base} N/class={args.n_per_class or 'full'} seed={args.seed} "
          f"split_seed={args.split_seed} | params={n_total} embed_params={n_embed} "
          f"lr={args.lr} epochs={args.epochs} bs={args.batch_size} | "
          f"train{tuple(tx.shape)} val_sel{tuple(vx.shape)} test{tuple(tex.shape)} "
          f"device={device}", flush=True)

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
        m = evaluate(model, vx, vy, device, C)
        if m["auc"] > best_sel_auc:
            best_sel_auc = m["auc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"[ep {ep+1:03d}] val_sel AUC={m['auc']:.4f} acc={m['acc']:.4f}", flush=True)

    model.load_state_dict(best_state)
    test_m = evaluate(model, tex, tey, device, C)

    R_drift = None
    if R_init is not None:
        with torch.no_grad():
            R_drift = (model.embed.refine_matrix().detach().cpu()
                       - R_init).norm().item()

    per_s = " ".join(f"{cn[i]}={test_m['auc_per'][i]:.4f}" for i in range(len(cn)))
    print(f"{'='*60}\nTEST (best val_sel epoch, held-out, evaluated once)\n{'='*60}")
    print(f"[DONE-QPE2] embed={args.embed} {base} N={args.n_per_class or 'full'} "
          f"seed={args.seed} test_AUC={test_m['auc']:.4f} val_sel_AUC={best_sel_auc:.4f} "
          f"per_class=({per_s}) params={n_total} embed_params={n_embed} "
          f"R_frob_drift={R_drift}", flush=True)

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
                batch_size=args.batch_size,
                R_frob_drift=R_drift)) + "\n")


if __name__ == "__main__":
    main()
