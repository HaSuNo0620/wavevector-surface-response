from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_summary(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("empty summary CSV")
    out: dict[str, np.ndarray] = {}
    for key in rows[0].keys():
        try:
            out[key] = np.asarray([float(r[key]) for r in rows], dtype=float)
        except ValueError:
            pass
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("summary", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = p.parse_args()

    d = read_summary(args.summary)
    order = np.argsort(d["qx"])
    q = d["qx"][order]
    total = d["var_total"][order]
    hh = d["var_hh"][order]
    cross2 = 2.0 * d["cross_h_perp"][order]
    pp = d["var_perpperp"][order]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    for key, label, marker in [
        ("capture_m2", "m=2", "o"),
        ("capture_m4", "m=4", "s"),
        ("capture_m8", "m=8", "^"),
        ("capture_m16", "m=16", "d"),
    ]:
        ax.plot(q, d[key][order], marker=marker, label=label)
    ax.set_xlabel(r"$q_\parallel$")
    ax.set_ylabel("capillary subspace capture fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "cumulative_capture_vs_q.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(q, d["principal1_m4"][order], marker="o", label=r"$\cos^2\theta_1$, m=4")
    ax.plot(q, d["principal2_m4"][order], marker="s", label=r"$\cos^2\theta_2$, m=4")
    ax.plot(q, d["principal1_m8"][order], marker="^", label=r"$\cos^2\theta_1$, m=8")
    ax.plot(q, d["principal2_m8"][order], marker="d", label=r"$\cos^2\theta_2$, m=8")
    ax.set_xlabel(r"$q_\parallel$")
    ax.set_ylabel("principal-angle squared cosine")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "principal_angles_vs_q.png", dpi=180)
    plt.close(fig)

    denom = np.where(np.abs(total) > 1e-30, total, np.nan)
    fig, ax = plt.subplots()
    ax.plot(q, hh / denom, marker="o", label="capillary subspace")
    ax.plot(q, cross2 / denom, marker="s", label="2 x cross")
    ax.plot(q, pp / denom, marker="^", label="orthogonal/packing")
    ax.set_xlabel(r"$q_\parallel$")
    ax.set_ylabel("fraction of total projected variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "sector_fractions_vs_q.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(q, d["profile_drift_rms"][order], marker="o")
    ax.set_xlabel(r"$q_\parallel$")
    ax.set_ylabel("first-half vs second-half profile RMS")
    fig.tight_layout()
    fig.savefig(args.output_dir / "profile_drift_rms.png", dpi=180)
    plt.close(fig)

    print(f"saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
