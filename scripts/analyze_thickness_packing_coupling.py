from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wvsr.analysis import two_interface_templates, decompose_subspace


def _raw_template_norms(rho0: np.ndarray, z: np.ndarray, centers: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho0, dtype=float)
    zz = np.asarray(z, dtype=float)
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0
    smooth = rho.copy()
    for _ in range(2):
        acc = np.zeros_like(smooth)
        for w, off in zip(kernel, np.arange(-2, 3)):
            acc += w * np.roll(smooth, int(off))
        smooth = acc
    deriv = -np.gradient(smooth, zz, edge_order=2)

    dz = float(np.mean(np.diff(zz)))
    lz = dz * len(zz)
    separation = min(abs(centers[1] - centers[0]), lz - abs(centers[1] - centers[0]))
    sigma = max(2.0 * dz, separation / 6.0)

    norms = []
    for center in centers:
        d = np.abs(zz - center)
        d = np.minimum(d, lz - d)
        window = np.exp(-0.5 * (d / sigma) ** 2)
        raw = deriv * window
        norms.append(np.sqrt(np.real(np.trapezoid(np.conj(raw) * raw, zz))))
    return np.asarray(norms, dtype=float)


def _center(a: np.ndarray) -> np.ndarray:
    return a - a.mean(axis=0, keepdims=True)


def _scalar_vector_metrics(s: np.ndarray, r_tz: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return cross-covariance profile, global normalized coupling and max pointwise corr."""
    ss = np.asarray(s, dtype=np.complex128) - np.mean(s)
    rr = _center(np.asarray(r_tz, dtype=np.complex128))
    cross_z = np.mean(ss[:, None] * np.conj(rr), axis=0)
    var_s = float(np.mean(np.abs(ss) ** 2).real)
    var_z = np.mean(np.abs(rr) ** 2, axis=0).real
    trace_r = float(np.trapezoid(var_z, z).real)
    cross_norm2 = float(np.trapezoid(np.abs(cross_z) ** 2, z).real)
    eta = cross_norm2 / (var_s * trace_r) if var_s > 0.0 and trace_r > 0.0 else np.nan
    denom = np.sqrt(np.maximum(var_s * var_z, 1e-300))
    max_corr = float(np.max(np.abs(cross_z) / denom))
    return cross_z, eta, max_corr


def _pc_explained_fraction(s: np.ndarray, r_tz: np.ndarray, k: int) -> float:
    """Variance fraction of scalar s linearly associated with first k residual PCs.

    This is a descriptive in-sample diagnostic; it can be upward biased for large k,
    so k=1,2,4,8 are reported separately instead of using all z bins.
    """
    ss = np.asarray(s, dtype=np.complex128) - np.mean(s)
    rr = _center(np.asarray(r_tz, dtype=np.complex128))
    n = len(ss)
    cov = (rr.conj().T @ rr) / max(n - 1, 1)
    cov = 0.5 * (cov + cov.conj().T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals.real)[::-1]
    vecs = vecs[:, order[: min(k, vecs.shape[1])]]
    scores = rr @ vecs
    var_s = float(np.mean(np.abs(ss) ** 2).real)
    if var_s <= 0.0:
        return np.nan
    covs = np.mean(ss[:, None] * np.conj(scores), axis=0)
    vars_score = np.mean(np.abs(scores) ** 2, axis=0).real
    return float(np.sum(np.abs(covs) ** 2 / np.maximum(var_s * vars_score, 1e-300)).real)


def main() -> None:
    p = argparse.ArgumentParser(description="Direct coupling between slab thickness W_q and packing residual")
    p.add_argument("input", type=Path, help="MC NPZ from scripts/run_mc.py")
    p.add_argument("--output-dir", type=Path, default=Path("results/processed/thickness_packing"))
    args = p.parse_args()

    d = np.load(args.input)
    z = d["z"]
    qx = d["qx"]
    rho0 = d["rho0_tz"].mean(axis=0)
    rho_all = d["rho_q_tqz"]

    templates, centers = two_interface_templates(rho0, z)
    raw_norms = _raw_template_norms(rho0, z, centers)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    profiles_W = []
    profiles_H = []

    for iq, q in enumerate(qx):
        rho = rho_all[:, iq, :]
        dec = decompose_subspace(rho, templates, z)
        h = dec.coefficients / raw_norms[None, :]
        h = _center(h)
        h1, h2 = h[:, 0], h[:, 1]
        H = 0.5 * (h1 + h2)
        W = h2 - h1
        residual = dec.rho_perp

        cross_W, eta_W, maxcorr_W = _scalar_vector_metrics(W, residual, z)
        cross_H, eta_H, maxcorr_H = _scalar_vector_metrics(H, residual, z)
        profiles_W.append(cross_W)
        profiles_H.append(cross_H)

        row = {
            "iq": iq,
            "qx": float(q),
            "var_W": float(np.mean(np.abs(W - W.mean()) ** 2).real),
            "var_H": float(np.mean(np.abs(H - H.mean()) ** 2).real),
            "eta_W_perp": eta_W,
            "eta_H_perp": eta_H,
            "max_point_corr_W_perp": maxcorr_W,
            "max_point_corr_H_perp": maxcorr_H,
        }
        for k in (1, 2, 4, 8):
            row[f"R2_W_pc{k}"] = _pc_explained_fraction(W, residual, k)
            row[f"R2_H_pc{k}"] = _pc_explained_fraction(H, residual, k)
        rows.append(row)

    csv_path = args.output_dir / "thickness_packing_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        args.output_dir / "thickness_packing_profiles.npz",
        qx=np.asarray(qx),
        z=np.asarray(z),
        cross_W_perp_qz=np.asarray(profiles_W),
        cross_H_perp_qz=np.asarray(profiles_H),
        interface_centers=np.asarray(centers),
    )

    q = np.asarray([r["qx"] for r in rows])
    etaW = np.asarray([r["eta_W_perp"] for r in rows])
    etaH = np.asarray([r["eta_H_perp"] for r in rows])
    r2W = np.asarray([r["R2_W_pc4"] for r in rows])
    r2H = np.asarray([r["R2_H_pc4"] for r in rows])
    mcW = np.asarray([r["max_point_corr_W_perp"] for r in rows])
    mcH = np.asarray([r["max_point_corr_H_perp"] for r in rows])

    fig, ax = plt.subplots()
    ax.plot(q, etaW, marker="o", label=r"$W$--residual")
    ax.plot(q, etaH, marker="s", label=r"$H$--residual")
    ax.set_xlabel(r"$q_\parallel$")
    ax.set_ylabel("global normalized cross-covariance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "global_coupling_vs_q.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(q, r2W, marker="o", label=r"$W$: first 4 residual PCs")
    ax.plot(q, r2H, marker="s", label=r"$H$: first 4 residual PCs")
    ax.set_xlabel(r"$q_\parallel$")
    ax.set_ylabel("in-sample explained fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "pc4_coupling_vs_q.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(q, mcW, marker="o", label=r"$W$")
    ax.plot(q, mcH, marker="s", label=r"$H$")
    ax.set_xlabel(r"$q_\parallel$")
    ax.set_ylabel("max pointwise |correlation|")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "max_pointwise_correlation_vs_q.png", dpi=180)
    plt.close(fig)

    print(f"saved: {csv_path}")
    for r in rows:
        print(
            f"q={r['qx']:.6f} etaW={r['eta_W_perp']:.5f} etaH={r['eta_H_perp']:.5f} "
            f"R2W4={r['R2_W_pc4']:.4f} R2H4={r['R2_H_pc4']:.4f} "
            f"maxW={r['max_point_corr_W_perp']:.4f}"
        )


if __name__ == "__main__":
    main()
