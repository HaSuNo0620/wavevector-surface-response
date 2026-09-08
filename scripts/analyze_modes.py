from __future__ import annotations

import argparse
from pathlib import Path
import csv

import numpy as np

from wvsr.analysis import (
    covariance_kernel,
    translational_template,
    diagonalize_covariance,
    template_overlaps,
    decompose_mode,
    sector_variances,
)


def analyze_one_q(z: np.ndarray, rho0: np.ndarray, rho: np.ndarray, n_modes: int) -> dict:
    cov = covariance_kernel(rho)
    phi = translational_template(rho0, z)
    eigvals, eigvecs = diagonalize_covariance(cov, n_modes)
    overlaps = template_overlaps(eigvecs, phi, z)

    dec = decompose_mode(rho, phi, z)
    dz = float(np.mean(np.diff(z)))
    total = (rho - rho.mean(axis=0, keepdims=True)).sum(axis=1) * dz
    h_obs = dec.rho_parallel.sum(axis=1) * dz
    p_obs = dec.rho_perp.sum(axis=1) * dz
    sectors = sector_variances(total, h_obs, p_obs)
    return {
        "covariance": cov,
        "template": phi,
        "eigenvalues": eigvals,
        "eigenvectors": eigvecs,
        "overlaps": overlaps,
        "h_t": dec.h,
        "sectors": sectors,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Goldstone/packing mode analysis of MC density modes")
    p.add_argument("input", type=Path, help="MC NPZ from scripts/run_mc.py")
    p.add_argument("--output-dir", type=Path, default=Path("results/processed/modes"))
    p.add_argument("--n-modes", type=int, default=12)
    args = p.parse_args()

    d = np.load(args.input)
    z = d["z"]
    qx = d["qx"]
    rho0 = d["rho0_tz"].mean(axis=0)
    rho_all = d["rho_q_tqz"]
    if rho_all.ndim != 3 or rho_all.shape[1] != len(qx):
        raise ValueError("expected rho_q_tqz with shape (time, q, z)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for iq, q in enumerate(qx):
        rho = rho_all[:, iq, :]
        out = analyze_one_q(z, rho0, rho, args.n_modes)
        sectors = out["sectors"]
        dominant = int(np.argmax(out["overlaps"]))
        np.savez_compressed(
            args.output_dir / f"q_{iq:03d}.npz",
            qx=q,
            z=z,
            rho0=rho0,
            template=out["template"],
            covariance=out["covariance"],
            eigenvalues=out["eigenvalues"],
            eigenvectors=out["eigenvectors"],
            overlap_rho_prime=out["overlaps"],
            h_t=out["h_t"],
            **sectors,
        )
        rows.append({
            "iq": iq,
            "qx": float(q),
            "dominant_goldstone_mode": dominant,
            "goldstone_overlap": float(out["overlaps"][dominant]),
            "goldstone_eigenvalue": float(out["eigenvalues"][dominant]),
            "largest_eigenvalue": float(out["eigenvalues"][0]),
            **sectors,
        })

    csv_path = args.output_dir / "mode_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {csv_path}")
    for row in rows:
        print(
            f"q={row['qx']:.6f} overlap={row['goldstone_overlap']:.4f} "
            f"lambda_G={row['goldstone_eigenvalue']:.6g} closure={row['closure_error']:.3e}"
        )


if __name__ == "__main__":
    main()
