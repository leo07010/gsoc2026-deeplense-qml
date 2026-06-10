#!/usr/bin/env python
"""
MAE pretrain (same recipe as the paper) → fine-tune with quantum / classical /
sham head, on a cached Model_X dataset (model_X.npz). Two stages:

  Stage 1 (--stage pretrain): self-supervised MAE on no_sub images (mask 0.9,
           patch 4) using mainv2's ViTEncoder + MaskedAutoencoderViT. Saves encoder.
  Stage 2 (--stage finetune --pretrained enc.pth --head {classical,quantum,sham}):
           load pretrained encoder, attach head, fine-tune on all 3 classes.

This puts quantum on the SAME winning recipe (pretrain + head) as the classical
MAE SOTA — the fair comparison the from-scratch runs lacked.
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

# ── quantum head pieces (8-qubit angle-encoded PQC) ──
N_Q, N_LAYERS, K = 8, 4, 8
_DEV = qml.device("default.qubit", wires=N_Q)
_QSHAPE = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_Q)


@qml.qnode(_DEV, interface="torch", diff_method="backprop")
def _circuit(angles, weights):
    for i in range(N_Q):
        qml.RY(angles[..., i], wires=i)
    qml.StronglyEntanglingLayers(weights, wires=range(N_Q))
    return [qml.expval(qml.PauliZ(i)) for i in range(K)]


def make_encoder():
    return ViTEncoder(img_size=64, patch_size=4, in_chans=1, embed_dim=192,
                      depth=6, num_heads=3, mlp_ratio=4.0, drop_rate=0.0)


class Head(nn.Module):
    def __init__(self, mode, in_dim=192, n_classes=3):
        super().__init__()
        self.mode = mode
        if mode == "classical":
            self.lin = nn.Linear(in_dim, n_classes)
        else:
            self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, N_Q if mode == "quantum" else K))
            if mode == "quantum":
                self.w = nn.Parameter(0.1 * torch.randn(_QSHAPE))
            self.out = nn.Sequential(nn.LayerNorm(K), nn.Linear(K, n_classes))

    def forward(self, cls):
        if self.mode == "classical":
            return self.lin(cls)
        z = self.proj(cls)
        if self.mode == "quantum":
            z = torch.stack(_circuit(torch.tanh(z) * np.pi, self.w), dim=-1).to(cls.dtype)
        else:                                    # sham
            z = torch.tanh(z)
        return self.out(z)


class Net(nn.Module):
    def __init__(self, encoder, mode, n_classes=3):
        super().__init__(); self.encoder = encoder; self.head = Head(mode, encoder.embed_dim, n_classes)
    def forward(self, x):
        return self.head(self.encoder(x)[:, 0])


def load_imgs(npz):
    d = np.load(npz, allow_pickle=True)
    return (torch.from_numpy(d["train_x"]).float().unsqueeze(1), torch.from_numpy(d["train_y"]).long(),
            torch.from_numpy(d["val_x"]).float().unsqueeze(1), torch.from_numpy(d["val_y"]).long(),
            list(d["class_names"]))


def evaluate(net, x, y, device, C, bs=256):
    net.eval(); lg = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            lg.append(net(x[i:i + bs].to(device)).cpu())
    p = torch.softmax(torch.cat(lg), 1).numpy(); yy = y.numpy()
    yb = label_binarize(yy, classes=np.arange(C))
    cm = confusion_matrix(yy, p.argmax(1))
    return (roc_auc_score(yb, p, average="macro", multi_class="ovr"),
            accuracy_score(yy, p.argmax(1)), f1_score(yy, p.argmax(1), average="macro"), cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--encoder", required=True, help="path to save/load encoder .pth")
    ap.add_argument("--head", choices=["classical", "quantum", "sham"], default="classical")
    ap.add_argument("--mae_epochs", type=int, default=15)
    ap.add_argument("--ft_epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tx, ty, vx, vy, cn = load_imgs(args.data); C = len(cn)
    nosub = cn.index("no_sub")

    if args.stage == "pretrain":
        enc = make_encoder()
        mae = MaskedAutoencoderViT(enc, img_size=64, patch_size=4, in_chans=1, mask_ratio=0.9)
        imgs = tx[ty == nosub]                      # pretrain on no_sub only (paper recipe)
        print(f"[PRETRAIN] MAE on {len(imgs)} no_sub imgs, mask 0.9, {args.mae_epochs} ep", flush=True)
        loader = DataLoader(imgs, batch_size=128, shuffle=True, num_workers=4)
        train_mae(mae, loader, device, epochs=args.mae_epochs, lr=1e-4)
        torch.save(enc.state_dict(), args.encoder)
        print(f"[PRETRAIN] saved encoder → {args.encoder}", flush=True)
        return

    # finetune
    enc = make_encoder(); enc.load_state_dict(torch.load(args.encoder, map_location=device))
    net = Net(enc, args.head, C).to(device)
    nparam = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"[FINETUNE] head={args.head} data={os.path.basename(args.data)} params={nparam} "
          f"train{tuple(tx.shape)} val{tuple(vx.shape)}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.ft_epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=128, shuffle=True)
    best = 0.0
    for ep in range(args.ft_epochs):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(net(xb), yb).backward(); opt.step()
        sched.step()
        auc, acc, f1, cm = evaluate(net, vx, vy, device, C); best = max(best, auc)
        ai = cn.index("axion") if "axion" in cn else 0
        print(f"[ep {ep+1:02d}] AUC={auc:.4f} acc={acc:.4f} f1={f1:.4f} axion={cm[ai,ai]/cm[ai].sum():.4f}",
              flush=True)
    print(f"\n[DONE] {args.head} on {os.path.basename(args.data)} best AUC={best:.4f} params={nparam}")


if __name__ == "__main__":
    main()
