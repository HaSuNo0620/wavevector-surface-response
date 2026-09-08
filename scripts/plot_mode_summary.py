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
    overlap = d["capillary_overlap"][order]
    overlap2 = d["second_capillary_overlap"][order]
    lam = d["capillary_eigenvalue"][order]
    lam2 = d["second_capillary_eigenvalue"][order]
    total = d["var_total"][order]
    hh = d["var_hh"][order]
    cross2 = 2.0 * d["cross_h_perp"][order]
    pp = d["var_perpperp"][order]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(q, overlap, marker="o", label="1st capillary-like mode")
    ax.plot(q, overlap2, marker="s", label="2nd capillary-like mode")
    ax.set_xlabel(r"$q_\parallel$")
    ax.set_ylabel("overlap with two-interface capillary subspace")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "capillary_overlap_vs_q.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(q, lam, marker="o", label="1st")
    ax.plot(q, lam2, marker="s", label="2nd")
    ax.set_xlabel(r"$q_\parallel$")
    ax.set_ylabel("capillary-like covariance eigenvalue")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "capillary_eigenvalue_vs_q.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(q, lam * q * q, marker="o", label="1st")
    ax.plot(q, lam2 * q * q, marker="s", label="2nd")
    ax.set_xlabel(r"$q_\parallel$")
    ax.set_ylabel(r"$q_\parallel^2\lambda_{cap}$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "capillary_q2lambda_vs_q.png", dpi=180)
    plt.close(fig)

    denom = np.where(np.abs(total) > 1e-30, total, np.nan)
    fig, ax = plt.subplots()
    ax.plot(q, hh / denom, marker="o", label="capillary subspace")
    ax.plot(q, cross2 / denom, marker="o", label="2 x cross")
    ax.plot(q, pp / denom, marker="o", label="orthogonal/packing")
    ax.set_xlabel(r"$q_\parallel$")
    ax.set_ylabel("fraction of total projected variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "sector_fractions_vs_q.png", dpi=180)
    plt.close(fig)

    print(f"saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
