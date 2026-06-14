#!/usr/bin/env python
"""
PAPER-FAITHFUL MAE pretrain → finetune (arXiv:2512.06642, Section 4 recipe),
replacing the recipe-mismatched pretrain_finetune.py:

                       v1 (mismatched)            v2 (paper Section 4)
  pretrain epochs      15                          10
  pretrain batch       128                         64
  pretrain optimizer   Adam lr 1e-4                Adam lr 1e-4, wd 0       (same)
  finetune optimizer   AdamW lr 5e-4 wd 1e-4       Adam  lr 5e-5 wd 1e-5
  finetune batch       128                         64
  finetune epochs      20 + cosine + label smooth  10, plain CE, no sched
  encoder dropout      0.0                         0.1 (finetune only)

Heads:
  classical : Linear(192→3)  — exactly the paper's ViTClassifier head
  quantum   : QVF-style neural amplitude encoding, NO information bottleneck:
              192 → energy MLP → 256 Boltzmann amplitudes → 8-qubit PQC
              → 8 ⟨Z⟩ → Linear(8→3). Circuit+NAE get their own (higher) lr.
  sham      : same NAE but classical Linear(256→8)+tanh replaces the circuit.

Stages: --stage pretrain | finetune. Data: model_X.npz from cache_model.py.
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if "jax" not in sys.modules:
    sys.modules["jax"] = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize

from mainv2 import ViTEncoder, MaskedAutoencoderViT, train_mae

N_Q, N_LAYERS, K = 8, 4, 8
DIM = 2 ** N_Q                                   # 256 amplitudes
_DEV = qml.device("default.qubit", wires=N_Q)
_QSHAPE = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_Q)


@qml.qnode(_DEV, interface="torch", diff_method="backprop")
def _circuit(amp, weights):
    qml.AmplitudeEmbedding(amp, wires=range(N_Q), normalize=True)
    qml.StronglyEntanglingLayers(weights, wires=range(N_Q))
    return [qml.expval(qml.PauliZ(i)) for i in range(K)]


def make_encoder(drop_rate=0.0):
    return ViTEncoder(img_size=64, patch_size=4, in_chans=1, embed_dim=192,
                      depth=6, num_heads=3, mlp_ratio=4.0, drop_rate=drop_rate)


class Head(nn.Module):
    def __init__(self, mode, in_dim=192, n_classes=3, e_hid=128):
        super().__init__()
        self.mode = mode
        if mode == "classical":
            self.lin = nn.Linear(in_dim, n_classes)         # paper head, verbatim
        else:
            # QVF neural amplitude encoding: full 192-d feature → energy manifold
            # → 256 Boltzmann amplitudes (no low-dim linear bottleneck)
            self.energy = nn.Sequential(nn.LayerNorm(in_dim),
                                        nn.Linear(in_dim, e_hid), nn.Tanh(),
                                        nn.Linear(e_hid, DIM))
            if mode == "quantum":
                self.w = nn.Parameter(0.1 * torch.randn(_QSHAPE))
            else:                                            # sham
                self.cl = nn.Linear(DIM, K)
            self.out = nn.Sequential(nn.LayerNorm(K), nn.Linear(K, n_classes))

    def quantum_params(self):
        if self.mode == "classical":
            return []
        ps = list(self.energy.parameters()) + list(self.out.parameters())
        ps += [self.w] if self.mode == "quantum" else list(self.cl.parameters())
        return ps

    def forward(self, cls):
        if self.mode == "classical":
            return self.lin(cls)
        amp = torch.sqrt(torch.softmax(-self.energy(cls), dim=1) + 1e-12)
        if self.mode == "quantum":
            z = torch.stack(_circuit(amp, self.w), dim=-1).to(cls.dtype)
        else:
            z = torch.tanh(self.cl(amp))
        return self.out(z)


class Net(nn.Module):
    def __init__(self, encoder, mode, n_classes=3):
        super().__init__()
        self.encoder = encoder
        self.head = Head(mode, encoder.embed_dim, n_classes)

    def forward(self, x):
        return self.head(self.encoder(x)[:, 0])


def load_imgs(npz):
    d = np.load(npz, allow_pickle=True)
    return (torch.from_numpy(d["train_x"]).float().unsqueeze(1),
            torch.from_numpy(d["train_y"]).long(),
            torch.from_numpy(d["val_x"]).float().unsqueeze(1),
            torch.from_numpy(d["val_y"]).long(),
            [str(c) for c in d["class_names"]])


def evaluate(net, x, y, device, C, bs=256):
    net.eval(); lg = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            lg.append(net(x[i:i + bs].to(device)).cpu())
    p = torch.softmax(torch.cat(lg), 1).numpy(); yy = y.numpy()
    yb = label_binarize(yy, classes=np.arange(C))
    cm = confusion_matrix(yy, p.argmax(1))
    return (roc_auc_score(yb, p, average="macro", multi_class="ovr"),
            accuracy_score(yy, p.argmax(1)),
            f1_score(yy, p.argmax(1), average="macro"), cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--encoder", required=True, help="path to save/load encoder .pth")
    ap.add_argument("--scratch", action="store_true",
                    help="finetune from random init (paper's 'Scratch' baseline)")
    ap.add_argument("--head", choices=["classical", "quantum", "sham"], default="classical")
    ap.add_argument("--mae_epochs", type=int, default=10)        # paper: 10
    ap.add_argument("--ft_epochs", type=int, default=10)         # paper: 10
    ap.add_argument("--lr", type=float, default=5e-5)            # paper: 5e-5
    ap.add_argument("--wd", type=float, default=1e-5)            # paper: 1e-5
    ap.add_argument("--qlr", type=float, default=1e-3,
                    help="lr for circuit/NAE params (paper lr is too small for PQCs)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tx, ty, vx, vy, cn = load_imgs(args.data); C = len(cn)
    nosub = cn.index("no_sub")
    if args.smoke:
        tx, ty, vx, vy = tx[:512], ty[:512], vx[:1024], vy[:1024]
        args.mae_epochs = args.ft_epochs = 2

    if args.stage == "pretrain":
        enc = make_encoder(drop_rate=0.0)
        mae = MaskedAutoencoderViT(enc, img_size=64, patch_size=4, in_chans=1,
                                   mask_ratio=0.9)                 # paper: 0.9
        imgs = tx[ty == nosub]
        print(f"[PRETRAIN-v2] paper recipe: {len(imgs)} no_sub imgs, mask 0.9, "
              f"{args.mae_epochs} ep, Adam 1e-4, batch 64", flush=True)
        loader = DataLoader(imgs, batch_size=64, shuffle=True, num_workers=4)
        train_mae(mae, loader, device, epochs=args.mae_epochs, lr=1e-4)
        torch.save(enc.state_dict(), args.encoder)
        print(f"[PRETRAIN-v2] saved encoder → {args.encoder}", flush=True)
        return

    # ── finetune, paper Section 4: Adam lr 5e-5 wd 1e-5, dropout 0.1, batch 64 ──
    enc = make_encoder(drop_rate=0.1)
    if not args.scratch:
        enc.load_state_dict(torch.load(args.encoder, map_location=device))
    net = Net(enc, args.head, C).to(device)
    nparam = sum(p.numel() for p in net.parameters() if p.requires_grad)
    qparams = net.head.quantum_params()
    qids = {id(p) for p in qparams}
    base = [p for p in net.parameters() if id(p) not in qids]
    groups = [{"params": base, "lr": args.lr}]
    if qparams:
        groups.append({"params": qparams, "lr": args.qlr})
    opt = torch.optim.Adam(groups, weight_decay=args.wd)
    crit = nn.CrossEntropyLoss()
    print(f"[FT-v2] head={args.head} data={os.path.basename(args.data)} "
          f"params={nparam} | Adam lr={args.lr} (q-group lr={args.qlr if qparams else '-'}) "
          f"wd={args.wd} dropout=0.1 batch=64 ep={args.ft_epochs}", flush=True)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=64, shuffle=True)
    best = 0.0
    for ep in range(args.ft_epochs):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(net(xb), yb).backward(); opt.step()
        auc, acc, f1, cm = evaluate(net, vx, vy, device, C); best = max(best, auc)
        ai = cn.index("axion") if "axion" in cn else 0
        print(f"[ep {ep+1:02d}] AUC={auc:.4f} acc={acc:.4f} f1={f1:.4f} "
              f"axion={cm[ai,ai]/max(cm[ai].sum(),1):.4f}", flush=True)
    print(f"\n[DONE-v2] {args.head} on {os.path.basename(args.data)} "
          f"final AUC={auc:.4f} best AUC={best:.4f} params={nparam}")


if __name__ == "__main__":
    main()
