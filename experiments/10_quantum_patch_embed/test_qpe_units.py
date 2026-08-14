#!/usr/bin/env python3
"""Unit tests for train_qpe.py -- run on CPU, seconds. Every test prints
PASS/FAIL; exits nonzero on any failure."""
import sys
import numpy as np
import torch

from train_qpe import (RYCNOTLayer, dct2_matrix, extract_patches,
                       PatchEmbed, QPEViT, D, PATCH_SIZE)

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


torch.manual_seed(0)

# 1. circuit matrix is orthogonal
layer = RYCNOTLayer(D)
U = layer.get_matrix()
err = (U.T @ U - torch.eye(D)).abs().max().item()
check("circuit orthogonality", err < 1e-5, f"max|U^T U - I|={err:.2e}")

# 2. all angles zero -> permutation matrix (validates CNOT-ring build)
with torch.no_grad():
    layer.angles.zero_()
U0 = layer.get_matrix()
is_perm = (torch.isclose(U0.abs().sum(0), torch.ones(D)).all() and
           torch.isclose(U0.abs().sum(1), torch.ones(D)).all() and
           bool(((U0 == 0) | (U0 == 1)).all()))
check("zero-angle circuit is a permutation matrix", is_perm)

# 3. DCT matrix orthogonal and matches scipy dctn
M1 = dct2_matrix()
D2 = torch.kron(M1, M1)
err = (D2.T @ D2 - torch.eye(D)).abs().max().item()
check("2D DCT orthogonality", err < 1e-5, f"err={err:.2e}")
try:
    from scipy.fft import dctn
    patch = np.random.default_rng(1).normal(size=(8, 8))
    ours  = (D2 @ torch.from_numpy(patch.reshape(-1)).float()).numpy()
    ref   = dctn(patch, norm="ortho").reshape(-1)
    err   = np.abs(ours - ref).max()
    check("2D DCT matches scipy dctn(norm='ortho')", err < 1e-4, f"err={err:.2e}")
except ImportError:
    print("[SKIP] scipy not available for DCT cross-check")

# 4. patch extraction: token t=r*8+c, pixel p=i*8+j
img = torch.zeros(1, 64, 64)
for r in range(8):
    for c in range(8):
        for i in range(8):
            for j in range(8):
                img[0, r * 8 + i, c * 8 + j] = (r * 8 + c) + (i * 8 + j) / 100.0
P = extract_patches(img)[0]                     # (64 tokens, 64 pixels)
ok = all(torch.isclose(P[t, p], torch.tensor(t + p / 100.0), atol=1e-5)
         for t in [0, 7, 8, 63] for p in [0, 7, 8, 63])
check("patch extraction bit-mapping (t=r*8+c, p=i*8+j)", ok)

# 5. scramble: deterministic per seed, differs from quantum, still orthogonal action
e_q = PatchEmbed("quantum", seed=42)
e_s1 = PatchEmbed("scramble", seed=42)
e_s2 = PatchEmbed("scramble", seed=42)
e_s3 = PatchEmbed("scramble", seed=43)
with torch.no_grad():
    for e in (e_s1, e_s2, e_s3):
        e.circ.angles.copy_(e_q.circ.angles)    # same circuit weights
x = torch.rand(2, 64, 64)
oq, o1, o2, o3 = e_q(x), e_s1(x), e_s2(x), e_s3(x)
check("scramble deterministic for same seed", torch.allclose(o1, o2))
check("scramble differs from quantum", not torch.allclose(oq, o1))
check("scramble differs across seeds", not torch.allclose(o1, o3))
n_in  = extract_patches(x).norm(dim=-1)
check("quantum embed preserves per-patch norm", torch.allclose(oq.norm(dim=-1), n_in, atol=1e-4))
check("scramble embed preserves per-patch norm", torch.allclose(o1.norm(dim=-1), n_in, atol=1e-4))

# 6. gradient flows to circuit angles through the full model
model = QPEViT(3, embed="quantum", seed=42)
out = model(torch.rand(4, 64, 64))
loss = torch.nn.functional.cross_entropy(out, torch.tensor([0, 1, 2, 0]))
loss.backward()
g = model.embed.circ.angles.grad
check("gradient reaches circuit angles", g is not None and g.abs().sum().item() > 0,
      f"|grad|={g.abs().sum().item():.2e}" if g is not None else "grad is None")

# 7. param counts per arm (48 angles + 1 scalar gain for circuit arms;
# 1 gain for dct; conv has its own 4,160)
print("\n--- param counts ---")
expected_embed = {"quantum": 49, "scramble": 49, "dct": 1, "conv": 4160}
for mode in ["quantum", "scramble", "dct", "conv"]:
    m = QPEViT(3, embed=mode, seed=42)
    tot = sum(p.numel() for p in m.parameters() if p.requires_grad)
    emb = sum(p.numel() for p in m.embed_parameters() if p.requires_grad)
    print(f"{mode:10s} total={tot:7d}  embed={emb}")
    check(f"embed params for {mode} == {expected_embed[mode]}",
          emb == expected_embed[mode], f"got {emb}")

# backbone params must be IDENTICAL across arms (only embed differs)
backbones = []
for mode in ["quantum", "scramble", "dct", "conv"]:
    m = QPEViT(3, embed=mode, seed=42)
    tot = sum(p.numel() for p in m.parameters() if p.requires_grad)
    emb = sum(p.numel() for p in m.embed_parameters() if p.requires_grad)
    backbones.append(tot - emb)
check("backbone params identical across all 4 arms", len(set(backbones)) == 1,
      str(backbones))

# 8. backbone INIT must be bit-identical across arms for the same seed
states = {}
for mode in ["quantum", "scramble", "dct", "conv"]:
    torch.manual_seed(123)
    m = QPEViT(3, embed=mode, seed=42)
    states[mode] = {k: v.clone() for k, v in m.state_dict().items()
                    if not k.startswith("embed.")}
ok = all(all(torch.equal(states["quantum"][k], states[mode][k])
             for k in states["quantum"])
         for mode in ["scramble", "dct", "conv"])
check("backbone init bit-identical across all 4 arms (same seed)", ok)

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("ALL UNIT TESTS PASSED")
