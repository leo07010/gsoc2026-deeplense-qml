#!/usr/bin/env python3
"""Unit tests for train_qpe2.py -- CPU, seconds. Exits nonzero on failure."""
import sys
import numpy as np
import torch

from train_qpe2 import (RYCNOTLayer, ButterflyLayer, CayleyLayer,
                        dct2_matrix, extract_patches, PatchEmbed2,
                        QPE2ViT, D)

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


torch.manual_seed(0)
I = torch.eye(D)

# 1. all refinement charts produce orthogonal matrices (perturbed away
# from init, since several of them start exactly at I)
for name, layer in [("rycnot", RYCNOTLayer(D)),
                    ("butterfly", ButterflyLayer(D)),
                    ("cayley", CayleyLayer(D)),
                    ("skew48", CayleyLayer(D, n_params=48))]:
    with torch.no_grad():
        for p in layer.parameters():
            p.uniform_(-0.5, 0.5)
    U = layer.get_matrix()
    err = (U.T @ U - I).abs().max().item()
    check(f"{name} orthogonality", err < 1e-4, f"err={err:.2e}")

# 2. all classical charts init EXACTLY at identity
check("butterfly init == I", torch.allclose(ButterflyLayer(D).get_matrix(), I, atol=1e-6))
check("cayley init == I",    torch.allclose(CayleyLayer(D).get_matrix(),    I, atol=1e-6))
check("skew48 init == I", torch.allclose(CayleyLayer(D, n_params=48).get_matrix(), I, atol=1e-6))

# 2b. skew48 has exactly 48 free params, and its coordinate choice is
# identical across instances/seeds (must be, or arms differ per run)
s1, s2 = CayleyLayer(D, n_params=48), CayleyLayer(D, n_params=48)
check("skew48 has exactly 48 params", s1.w.numel() == 48, f"got {s1.w.numel()}")
check("skew48 coordinates are deterministic", torch.equal(s1._iu, s2._iu))
with torch.no_grad():
    s1.w.uniform_(-0.5, 0.5)
n_offdiag = int(((s1.get_matrix() - I).abs() > 1e-6).sum())
check("skew48 actually moves off I", n_offdiag > 0, f"{n_offdiag} entries changed")

# 2c. THE init that ships: quantum angles are exactly zero, and gradients
# at that point are nonzero for all 48 angles (this is what makes the
# zero-init safe -- verified here, not assumed)
q = RYCNOTLayer(D)
check("quantum angles init exactly 0", bool((q.angles == 0).all()))
Uq = q.get_matrix()
loss = (Uq * torch.randn(D, D, generator=torch.Generator().manual_seed(3))).sum()
loss.backward()
nz = int((q.angles.grad.abs() > 1e-12).sum())
check("all 48 quantum angles have nonzero grad at theta=0", nz == 48, f"{nz}/48")

# 3. rycnot at zero angles == the CNOT-ring permutation (a relabeling, det +1)
lay = RYCNOTLayer(D)
with torch.no_grad():
    lay.angles.zero_()
U0 = lay.get_matrix()
is_perm = (torch.isclose(U0.abs().sum(0), torch.ones(D)).all() and
           torch.isclose(U0.abs().sum(1), torch.ones(D)).all() and
           bool(((U0 == 0) | (U0 == 1)).all()))
check("rycnot zero-angle == permutation (channel relabeling)", is_perm)

# 4. composed embeds at init: dctfix/butterfly/cayley identical outputs;
#    quantum output == a row-permutation of them
x = torch.rand(2, 64, 64)
outs = {}
for mode in ["dctfix", "butterfly", "cayley", "skew48", "quantum"]:
    torch.manual_seed(1)
    outs[mode] = PatchEmbed2(mode)(x)     # ships-as-is init, no tweaking
check("dctfix == butterfly at init", torch.allclose(outs["dctfix"], outs["butterfly"], atol=1e-5))
check("dctfix == cayley at init",    torch.allclose(outs["dctfix"], outs["cayley"],    atol=1e-5))
check("dctfix == skew48 at init",    torch.allclose(outs["dctfix"], outs["skew48"],    atol=1e-5))
# quantum at its shipped init = P @ dct -> exact channel permutation of
# the dctfix output (checked exactly via the permutation matrix, not just
# via sorted values, which would only be a necessary condition)
torch.manual_seed(1)
P = PatchEmbed2("quantum").refine.get_matrix()
check("quantum(init) == dctfix composed with P (exact)",
      torch.allclose(outs["quantum"], outs["dctfix"] @ P.T, atol=1e-5))

# 5. norm preservation through composed embed (gain=1 at init)
p = extract_patches(x)
for mode in ["dctfix", "quantum", "skew48", "butterfly", "cayley"]:
    torch.manual_seed(1)
    e = PatchEmbed2(mode)
    ok = torch.allclose(e(x).norm(dim=-1), p.norm(dim=-1), atol=1e-4)
    check(f"{mode} embed preserves per-patch norm", ok)

# 6. gradients flow to every refinement's params through the full model
for mode in ["quantum", "skew48", "butterfly", "cayley"]:
    m = QPE2ViT(3, embed=mode)
    out = m(torch.rand(4, 64, 64))
    loss = torch.nn.functional.cross_entropy(out, torch.tensor([0, 1, 2, 0]))
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in m.embed.refine.parameters())
    check(f"grad reaches {mode} refinement", g > 0, f"|grad|={g:.2e}")

# 7. param counts
expected = {"quantum": 49, "skew48": 49, "butterfly": 193, "cayley": 2017,
            "dctfix": 1, "conv": 4160}
print("\n--- param counts ---")
backbones = []
for mode, exp in expected.items():
    m = QPE2ViT(3, embed=mode)
    tot = sum(p.numel() for p in m.parameters() if p.requires_grad)
    emb = sum(p.numel() for p in m.embed_parameters() if p.requires_grad)
    print(f"{mode:10s} total={tot:7d}  embed={emb}")
    check(f"embed params for {mode} == {exp}", emb == exp, f"got {emb}")
    backbones.append(tot - emb)
check("backbone params identical across all 6 arms", len(set(backbones)) == 1,
      str(backbones))

# 8. backbone init bit-identical across arms for the same seed
states = {}
for mode in expected:
    torch.manual_seed(123)
    m = QPE2ViT(3, embed=mode)
    states[mode] = {k: v.clone() for k, v in m.state_dict().items()
                    if not k.startswith("embed.")}
ok = all(all(torch.equal(states["quantum"][k], states[mode][k])
             for k in states["quantum"]) for mode in expected)
check("backbone init bit-identical across all 6 arms (same seed)", ok)

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("ALL UNIT TESTS PASSED")
