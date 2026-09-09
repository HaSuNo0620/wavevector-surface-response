from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wvsr.analysis import two_interface_templates, decompose_subspace
from scripts.analyze_capillary_coordinates import _raw_template_norms


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze slab translation and thickness modes")
    p.add_argument("input", type=Path, help="MC NPZ from scripts/run_mc.py")
    p.add_argument("--output-dir", type=Path, default=Path("results/processed/translation_thickness"))
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
        h = dec.coefficients / raw_norms[None, :]
        h = h - h.mean(axis=0, keepdims=True)
        h1, h2 = h[:, 0], h[:, 1]

        # Physical coordinates of the two-interface slab:
        # H : center/translation mode, W : thickness/breathing mode.
        H = 0.5 * (h1 + h2)
        W = h2 - h1

        var_H = float(np.mean(np.abs(H) ** 2).real)
        var_W = float(np.mean(np.abs(W) ** 2).real)
        cov_HW = np.mean(H * np.conj(W))
        corr_HW = (
            float(np.real(cov_HW) / np.sqrt(var_H * var_W))
            if var_H > 0.0 and var_W > 0.0 else np.nan
        )

        rows.append({
            "iq": iq,
            "qx": float(q),
            "var_H": var_H,
            "var_W": var_W,
            "cov_HW_re": float(np.real(cov_HW)),
            "cov_HW_im": float(np.imag(cov_HW)),
            "corr_HW_re": corr_HW,
            "q2_var_H": float(q * q * var_H),
            "q2_var_W": float(q * q * var_W),
            "thickness_to_translation_ratio": float(var_W / var_H) if var_H > 0.0 else np.nan,
            "template_norm_1": float(raw_norms[0]),
            "template_norm_2": float(raw_norms[1]),
            "interface_z_1": float(centers[0]),
            "interface_z_2": float(centers[1]),
        })

    csv_path = args.output_dir / "translation_thickness_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    q = np.asarray([r["qx"] for r in rows])
    vH = np.asarray([r["var_H"] for r in rows])
    vW = np.asarray([r["var_W"] for r in rows])
    corr = np.asarray([r["corr_HW_re"] for r in rows])
    ratio = np.asarray([r["thickness_to_translation_ratio"] for r in rows])

    fig, ax = plt.subplots()
    ax.loglog(q, vH, marker="o", label=r"$\langle |H_q|^2\rangle$")
    ax.loglog(q, vW, marker="s", label=r"$\langle |W_q|^2\rangle$")
    ax.set_xlabel(r"$q_\parallel$")
    ax.set_ylabel("mode variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "translation_thickness_variance_vs_q.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(q, q * q * vH, marker="o", label=r"$q^2\langle |H_q|^2\rangle$")
    ax.plot(q, q * q * vW, marker="s", label=r"$q^2\langle |W_q|^2\rangle$")
    ax.set_xlabel(r"$q_\parallel$")
    ax.set_ylabel(r"$q_\parallel^2$ weighted variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "translation_thickness_q2variance_vs_q.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(q, ratio, marker="o")
    ax.set_xlabel(r"$q_\parallel$")
    ax.set_ylabel(r"$\langle |W_q|^2\rangle / \langle |H_q|^2\rangle$")
    fig.tight_layout()
    fig.savefig(args.output_dir / "thickness_translation_ratio_vs_q.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(q, corr, marker="o")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel(r"$q_\parallel$")
    ax.set_ylabel(r"Re corr$(H_q,W_q)$")
    fig.tight_layout()
    fig.savefig(args.output_dir / "translation_thickness_correlation_vs_q.png", dpi=180)
    plt.close(fig)

    print(f"saved: {csv_path}")
    for r in rows:
        print(
            f"q={r['qx']:.6f} var_H={r['var_H']:.6g} var_W={r['var_W']:.6g} "
            f"ratio={r['thickness_to_translation_ratio']:.4f} corr={r['corr_HW_re']:.4f}"
        )


if __name__ == "__main__":
    main()
