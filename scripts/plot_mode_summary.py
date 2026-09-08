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
    keys = rows[0].keys()
    out: dict[str, np.ndarray] = {}
    for key in keys:
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
    q = d["qx"]
    order = np.argsort(q)
    q = q[order]
    overlap = d["goldstone_overlap"][order]
    lam = d["goldstone_eigenvalue"][order]
    total = d["var_total"][order]
    hh = d["var_hh"][order]
    cross2 = 2.0 * d["cross_h_perp"][order]
    pp = d["var_perpperp"][order]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(q, overlap, marker="o")
    ax.set_xlabel(r"$q_\parallel$")
    ax.set_ylabel(r"Goldstone overlap $O_G$")
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(args.output_dir / "goldstone_overlap_vs_q.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(q, lam, marker="o", label=r"$\lambda_G$")
    ax.set_xlabel(r"$q_\parallel$")
    ax.set_ylabel(r"Goldstone eigenvalue $\lambda_G$")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(args.output_dir / "goldstone_eigenvalue_vs_q.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(q, lam * q * q, marker="o")
    ax.set_xlabel(r"$q_\parallel$")
    ax.set_ylabel(r"$q_\parallel^2\lambda_G$")
    fig.tight_layout()
    fig.savefig(args.output_dir / "goldstone_q2lambda_vs_q.png", dpi=180)
    plt.close(fig)

    # Sector fractions are diagnostic only when var_total is well away from zero.
    denom = np.where(np.abs(total) > 1e-30, total, np.nan)
    fig, ax = plt.subplots()
    ax.plot(q, hh / denom, marker="o", label=r"$hh$")
    ax.plot(q, cross2 / denom, marker="o", label=r"$2h\perp$")
    ax.plot(q, pp / denom, marker="o", label=r"$\perp\perp$")
    ax.set_xlabel(r"$q_\parallel$")
    ax.set_ylabel("fraction of total projected variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "sector_fractions_vs_q.png", dpi=180)
    plt.close(fig)

    print(f"saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
