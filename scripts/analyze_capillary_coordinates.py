from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wvsr.analysis import two_interface_templates, decompose_subspace


def _raw_template_norms(rho0: np.ndarray, z: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Reconstruct norms of the unnormalized local displacement templates."""
    rho = np.asarray(rho0, dtype=float)
    zz = np.asarray(z, dtype=float)

    # Keep this in sync with two_interface_templates().
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


def main() -> None:
    p = argparse.ArgumentParser(description="Direct two-interface capillary-coordinate analysis")
    p.add_argument("input", type=Path, help="MC NPZ from scripts/run_mc.py")
    p.add_argument("--output-dir", type=Path, default=Path("results/processed/capillary_coordinates"))
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

    for iq, q in enumerate(qx):
        rho = rho_all[:, iq, :]
        dec = decompose_subspace(rho, templates, z)

        # decompose_subspace coefficients multiply normalized templates.
        # Physical displacement amplitudes multiply the unnormalized
        # -rho0'(z) local templates, hence h = c / ||phi_raw||.
        h = dec.coefficients / raw_norms[None, :]
        h = h - h.mean(axis=0, keepdims=True)
        h1, h2 = h[:, 0], h[:, 1]
        h_plus = (h1 + h2) / np.sqrt(2.0)
        h_minus = (h1 - h2) / np.sqrt(2.0)

        v1 = float(np.mean(np.abs(h1) ** 2).real)
        v2 = float(np.mean(np.abs(h2) ** 2).real)
        cross = np.mean(h1 * np.conj(h2))
        vplus = float(np.mean(np.abs(h_plus) ** 2).real)
        vminus = float(np.mean(np.abs(h_minus) ** 2).real)
        corr_re = float(np.real(cross) / np.sqrt(v1 * v2)) if v1 > 0 and v2 > 0 else np.nan

        rows.append({
            "iq": iq,
            "qx": float(q),
            "var_h1": v1,
            "var_h2": v2,
            "cov_h1_h2_re": float(np.real(cross)),
            "cov_h1_h2_im": float(np.imag(cross)),
            "corr_h1_h2_re": corr_re,
            "var_h_plus": vplus,
            "var_h_minus": vminus,
            "q2_var_h_plus": float(q * q * vplus),
            "q2_var_h_minus": float(q * q * vminus),
            "template_norm_1": float(raw_norms[0]),
            "template_norm_2": float(raw_norms[1]),
            "interface_z_1": float(centers[0]),
            "interface_z_2": float(centers[1]),
        })

    csv_path = args.output_dir / "capillary_coordinate_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    q = np.asarray([r["qx"] for r in rows])
    vp = np.asarray([r["var_h_plus"] for r in rows])
    vm = np.asarray([r["var_h_minus"] for r in rows])
    cp = np.asarray([r["corr_h1_h2_re"] for r in rows])

    fig, ax = plt.subplots()
    ax.loglog(q, vp, marker="o", label=r"$\langle |h_+|^2\rangle$")
    ax.loglog(q, vm, marker="s", label=r"$\langle |h_-|^2\rangle$")
    ref = vp[0] * (q[0] / q) ** 2
    ax.loglog(q, ref, linestyle="--", label=r"$q^{-2}$ reference")
    ax.set_xlabel(r"$q_\parallel$")
    ax.set_ylabel("capillary-coordinate variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "capillary_variance_vs_q.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(q, q * q * vp, marker="o", label=r"$q^2\langle |h_+|^2\rangle$")
    ax.plot(q, q * q * vm, marker="s", label=r"$q^2\langle |h_-|^2\rangle$")
    ax.set_xlabel(r"$q_\parallel$")
    ax.set_ylabel(r"$q_\parallel^2\langle |h_\pm|^2\rangle$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "capillary_q2variance_vs_q.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(q, cp, marker="o")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel(r"$q_\parallel$")
    ax.set_ylabel(r"Re corr$(h_1,h_2)$")
    fig.tight_layout()
    fig.savefig(args.output_dir / "interface_correlation_vs_q.png", dpi=180)
    plt.close(fig)

    print(f"saved: {csv_path}")
    for r in rows:
        print(
            f"q={r['qx']:.6f} var_plus={r['var_h_plus']:.6g} "
            f"var_minus={r['var_h_minus']:.6g} "
            f"q2plus={r['q2_var_h_plus']:.6g} q2minus={r['q2_var_h_minus']:.6g} "
            f"corr={r['corr_h1_h2_re']:.4f}"
        )


if __name__ == "__main__":
    main()
