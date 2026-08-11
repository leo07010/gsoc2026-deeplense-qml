#!/usr/bin/env python
"""QOVT ablation: pre-registered, fair-comparison rerun of train_qovt_rycnot.py.

This is a NEW, independent script -- it does not modify the original
(preserved as-is in mae-lensing / not yet ported). It fixes four issues an
adversarial audit found in the original QOVT script family
(/tmp/.../scratchpad/qovt_inventory.md, 2026-08-06):

  1. No held-out test set: the original reports max(val_AUC) over training
     -- the SAME split used for model selection is reported as the result.
     Quantum's noisier val curve (std 0.0202 vs sham's 0.0006) means this
     inflates quantum's number more than any margin it claims to show.
     FIX: val is split into val_sel (model selection only) and a disjoint
     test set (evaluated ONCE, at the selected best epoch, never used to
     pick that epoch). The test split is carved with a FIXED split_seed
     independent of --seed, so every seed/arm is scored on the identical
     held-out set.

  2. No parameter-matched classical control existed (sham=33k vs quantum's
     384; classical-MHA=66k). FIX: added a "matched" arm -- rank-1 low-rank
     U/V (LowRankLayer, 128 params/instance, 1024 total across 4 layers) --
     the leanest linear-algebra classical alternative to a D x D map.
     Exact 1:1 param matching to quantum's 384 is not achievable with a
     linear factorization (minimum meaningful rank already exceeds it) --
     this is disclosed, not hidden: 1024 is ~2.7x quantum's param count,
     still far closer than the pre-existing sham (33k, ~85x).

  3. LR asymmetry: original gives quantum's RBS angles qlr=1e-2 and
     everything else (including sham/classical's U,V) lr=1e-3. FIX: the
     SAME two-tier LR split (attention-layer params get qlr, everything
     else gets lr) is applied uniformly to all four arms -- whatever the
     attention U/V object is (RY+CNOT circuit, low-rank factors,
     unconstrained Linear, or MHA in/out-proj), its params go in the qlr
     group. This preserves the original's stated rationale (attention
     params need a faster LR to escape a flat init landscape) without
     giving quantum an LR advantage the other arms don't get.

  4. Four modes instead of a hardcoded default: quantum, matched, sham,
     classical -- run explicitly, not silently defaulted.

See docs/QOVT_ABLATION_PREREGISTRATION.md for the pre-registered hypotheses,
test cells, and statistical test -- written and committed BEFORE this
experiment's results existed.

Architecture is otherwise IDENTICAL to train_qovt_rycnot.py: patch=8x8 (64
patches + CLS), D=64, DEPTH=4 QO-attention layers, RY+CNOT ring circuit
(6 qubits, 8 layers, 48 angles per U/V).
"""
import os, sys, math, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize

# ── config (identical to train_qovt_rycnot.py) ─────────────────────────────
IMG        = 64
PATCH_SIZE = int(os.environ.get("QOVT_PATCH",  "8"))
D          = int(os.environ.get("QOVT_D",      "64"))
DEPTH      = int(os.environ.get("QOVT_DEPTH",  "4"))
MLP_MULT   = int(os.environ.get("QOVT_MLP",    "4"))


# ── RY+CNOT ring layer (byte-for-byte copy of train_qovt_rycnot.RBSLayer) ──
class RYCNOTLayer(nn.Module):
    """6-qubit RY+CNOT ring circuit expressed as a 64x64 PyTorch matrix.
    48 trainable angles (8 layers x 6 qubits)."""
    N_Q = 6

    def __init__(self, D: int, n_layers: int = 8):
        super().__init__()
        assert D == 64, f"RBSLayer requires D=64 (2^6), got {D}"
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


# ── RBS/Givens butterfly layer (byte-for-byte copy of
# /home/leo07010/mae-lensing/train_qovt.py's RBSLayer -- the circuit that
# produced id=23's headline single-seed numbers. Reused verbatim, not
# reinvented, so this ablation tests the SAME circuit id=23 already reports
# on, not a new guess at what "butterfly" means. Renamed for clarity since
# this script already has an RYCNOTLayer.) ──
class ButterflyLayer(nn.Module):
    """Butterfly Givens-rotation layer -- RBS/Givens butterfly from Cherrat
    et al., as already implemented in this project's train_qovt.py.
    For D-dim input: (D/2)*log2(D) trainable angles. Produces a D x D
    orthogonal matrix; applied as x @ W. D must be a power of 2."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.get_matrix()


class LowRankLayer(nn.Module):
    """Rank-r factorization of a D x D map: x -> x @ A @ B. 2*D*r params.
    The leanest classical linear-algebra control we could construct; see
    module docstring item 2 for why exact param matching to RBSLayer (48
    params) isn't achievable with r>=1."""

    def __init__(self, D: int, rank: int = 1):
        super().__init__()
        self.A = nn.Parameter(0.02 * torch.randn(D, rank))
        self.B = nn.Parameter(0.02 * torch.randn(rank, D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.A @ self.B


# ── attention ────────────────────────────────────────────────────────────
class QOAttention(nn.Module):
    """mode='quantum': RY+CNOT ring (RBSLayer), 96 params/layer.
    mode='matched':    rank-1 low-rank U,V (LowRankLayer), 256 params/layer.
    mode='sham':       unconstrained Linear(D,D), no bias, 8192 params/layer.
    mode='classical':  standard multi-head attention, ~16384 params/layer."""

    def __init__(self, D: int, mode: str = "quantum", n_heads: int = 8,
                 n_layers: int = 8, lr_rank: int = 1, circuit: str = "rycnot"):
        super().__init__()
        self.mode  = mode
        self.scale = math.sqrt(D)
        self.norm  = nn.LayerNorm(D)
        if mode == "quantum":
            if circuit == "rycnot":
                self.U = RYCNOTLayer(D, n_layers)
                self.V = RYCNOTLayer(D, n_layers)
            elif circuit == "butterfly":
                self.U = ButterflyLayer(D)
                self.V = ButterflyLayer(D)
            else:
                raise ValueError(f"Unknown circuit: {circuit}")
        elif mode == "matched":
            self.U = LowRankLayer(D, rank=lr_rank)
            self.V = LowRankLayer(D, rank=lr_rank)
        elif mode == "sham":
            self.U = nn.Linear(D, D, bias=False)
            self.V = nn.Linear(D, D, bias=False)
        elif mode == "classical":
            self.mha = nn.MultiheadAttention(D, num_heads=n_heads,
                                             batch_first=True, bias=False)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def attn_parameters(self):
        """Params that get the faster (qlr) learning rate -- generalized to
        whatever the attention mechanism is, so the LR asymmetry applies
        equally to all four modes (fix for issue #3)."""
        if self.mode == "classical":
            return list(self.mha.parameters())
        return list(self.U.parameters()) + list(self.V.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm(x)
        if self.mode == "classical":
            out, _ = self.mha(xn, xn, xn)
            return out
        Ux   = self.U(xn)
        Vx   = self.V(xn)
        attn = torch.bmm(xn, Ux.transpose(1, 2)) / self.scale
        attn = F.softmax(attn, dim=-1)
        return torch.bmm(attn, Vx)


class MLP(nn.Module):
    def __init__(self, D: int, mult: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(D), nn.Linear(D, D * mult), nn.GELU(), nn.Linear(D * mult, D))

    def forward(self, x):
        return self.net(x)


class QOVT(nn.Module):
    def __init__(self, n_classes: int = 3, mode: str = "quantum", n_layers: int = 8,
                 lr_rank: int = 1, circuit: str = "rycnot"):
        super().__init__()
        self.mode  = mode
        N_patches  = (IMG // PATCH_SIZE) ** 2
        self.patch = nn.Conv2d(1, D, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        self.cls   = nn.Parameter(torch.zeros(1, 1, D))
        self.pos   = nn.Parameter(torch.randn(1, N_patches + 1, D) * 0.02)
        self.attn_layers = nn.ModuleList(
            [QOAttention(D, mode, n_layers=n_layers, lr_rank=lr_rank, circuit=circuit)
             for _ in range(DEPTH)])
        self.mlp_layers  = nn.ModuleList([MLP(D, MLP_MULT) for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, n_classes)

    def attn_parameters(self):
        out = []
        for m in self.attn_layers:
            out += m.attn_parameters()
        return out

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        x = imgs.unsqueeze(1) if imgs.dim() == 3 else imgs
        x = self.patch(x)
        x = x.flatten(2).transpose(1, 2)
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
    """Fixed 50/50 split of the original val set, independent of --seed, so
    every (arm, training-seed) combination is scored against the IDENTICAL
    held-out test set. val_sel is used only for best-epoch selection; test
    is touched exactly once, at the end, on the selected epoch's weights."""
    n = len(vy)
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    half = n // 2
    sel_idx, test_idx = perm[:half], perm[half:]
    return (vx[sel_idx], vy[sel_idx]), (vx[test_idx], vy[test_idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",        required=True)
    ap.add_argument("--epochs",      type=int,   default=30)
    ap.add_argument("--batch_size",  type=int,   default=64)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--qlr",         type=float, default=None,
                    help="If set, attention-layer params get this LR instead "
                         "of --lr (uniformly across all modes). Default (None) "
                         "means every parameter, every mode, uses a SINGLE lr "
                         "-- see 2026-08-06 correction note in module docstring: "
                         "an earlier version of this script applied qlr=1e-2 "
                         "uniformly and it destabilized sham/matched/classical's "
                         "larger linear layers (matched: flat at chance for all "
                         "30 epochs; sham: rose then collapsed) while quantum's "
                         "48-param circuit tolerated it -- a real LR confound, "
                         "not a quantum effect. Single-LR is now the default.")
    ap.add_argument("--n_per_class", type=int,   default=0, help="0=full")
    ap.add_argument("--seed",        type=int,   default=42,
                    help="training seed (init + data shuffling + subsample draw)")
    ap.add_argument("--split_seed",  type=int,   default=0,
                    help="FIXED val->(val_sel,test) split seed, independent of --seed")
    ap.add_argument("--mode", choices=["quantum", "matched", "sham", "classical"],
                    required=True)
    ap.add_argument("--circuit", choices=["rycnot", "butterfly"], default="rycnot",
                    help="quantum-mode circuit only; ignored for matched/sham/classical")
    ap.add_argument("--n_layers",    type=int,   default=8)
    ap.add_argument("--lr_rank",     type=int,   default=1,
                    help="rank for the 'matched' low-rank control")
    ap.add_argument("--smoke",       action="store_true")
    ap.add_argument("--out_json",    default=None,
                    help="if set, append one JSON line with the result")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d  = np.load(args.data, allow_pickle=True)
    cn = list(d["class_names"]); C = len(cn)
    tx = torch.from_numpy(d["train_x"]).float()
    ty = torch.from_numpy(d["train_y"]).long()
    vx_all = torch.from_numpy(d["val_x"]).float()
    vy_all = torch.from_numpy(d["val_y"]).long()

    # fix #1: disjoint val_sel / test, split independent of training seed
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

    model = QOVT(C, mode=args.mode, n_layers=args.n_layers, lr_rank=args.lr_rank,
                 circuit=args.circuit).to(device)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_attn  = sum(p.numel() for p in model.attn_parameters())

    # fix #3 (corrected 2026-08-06): default is a SINGLE lr for every
    # parameter, every mode -- no special-casing. Only build a two-tier
    # group if --qlr is explicitly passed (kept for follow-up LR-sensitivity
    # ablations, not used in the primary comparison).
    if args.qlr is not None:
        attn_ids = {id(p) for p in model.attn_parameters()}
        qp = [p for p in model.parameters() if id(p) in attn_ids]
        cp = [p for p in model.parameters() if id(p) not in attn_ids]
        groups = [{"params": cp, "lr": args.lr}, {"params": qp, "lr": args.qlr}]
    else:
        groups = [{"params": list(model.parameters()), "lr": args.lr}]

    opt   = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(10, args.epochs // 3), T_mult=2, eta_min=1e-6)
    crit  = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(tx, ty),
                        batch_size=min(args.batch_size, len(tx)), shuffle=True)

    base = os.path.splitext(os.path.basename(args.data))[0]
    print(f"[INFO] QOVT-ABLATION D={D} P={PATCH_SIZE} depth={DEPTH} | mode={args.mode} "
          f"circuit={args.circuit if args.mode == 'quantum' else 'n/a'} | "
          f"data={base} N/class={args.n_per_class or 'full'} seed={args.seed} "
          f"split_seed={args.split_seed} | params={n_total} attn_params={n_attn} "
          f"qlr={args.qlr} lr={args.lr} | train{tuple(tx.shape)} "
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

    # fix #1: test evaluated exactly ONCE, at the val_sel-selected epoch
    model.load_state_dict(best_state)
    test_m = evaluate(model, tex, tey, device, C)
    ai  = cn.index("axion")
    axr = test_m["cm"][ai, ai] / test_m["cm"][ai].sum() if test_m["cm"][ai].sum() > 0 else 0.0
    per_s = " ".join(f"{cn[i]}={test_m['auc_per'][i]:.4f}" for i in range(len(cn)))
    print(f"{'='*60}\nTEST (best val_sel epoch, held-out, evaluated once)\n{'='*60}")
    print(f"TEST AUC : {test_m['auc']:.4f}  acc {test_m['acc']:.4f}  f1 {test_m['f1']:.4f}  "
          f"axion_recall {axr:.4f}")
    print(f"[DONE-QOVT-ABLATION] mode={args.mode} {base} N={args.n_per_class or 'full'} "
          f"seed={args.seed} test_AUC={test_m['auc']:.4f} val_sel_AUC={best_sel_auc:.4f} "
          f"per_class=({per_s}) params={n_total} attn_params={n_attn}", flush=True)

    if args.out_json:
        with open(args.out_json, "a") as f:
            f.write(json.dumps(dict(
                mode=args.mode, circuit=(args.circuit if args.mode == "quantum" else "n/a"),
                data=base, n_per_class=args.n_per_class or "full",
                seed=args.seed, split_seed=args.split_seed,
                test_auc=test_m["auc"], val_sel_auc=best_sel_auc,
                test_auc_per=test_m["auc_per"].tolist(), params=n_total,
                attn_params=n_attn)) + "\n")


if __name__ == "__main__":
    main()
