from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from wvsr.analysis import (
    covariance_kernel,
    translational_template,
    diagonalize_covariance,
    template_overlaps,
    decompose_mode,
    sector_variances,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path, help="NPZ with z, rho0, rho_q_tz")
    p.add_argument("--output", type=Path, default=Path("results/mode_analysis.npz"))
    p.add_argument("--n-modes", type=int, default=12)
    args = p.parse_args()

    d = np.load(args.input)
    z = d["z"]
    rho0 = d["rho0"]
    rho = d["rho_q_tz"]

    cov = covariance_kernel(rho)
    phi = translational_template(rho0, z)
    eigvals, eigvecs = diagonalize_covariance(cov, args.n_modes)
    overlaps = template_overlaps(eigvecs, phi, z)

    dec = decompose_mode(rho, phi, z)
    total = (rho - rho.mean(axis=0, keepdims=True)).sum(axis=1)
    h_obs = dec.rho_parallel.sum(axis=1)
    p_obs = dec.rho_perp.sum(axis=1)
    sectors = sector_variances(total, h_obs, p_obs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        z=z,
        rho0=rho0,
        template=phi,
        covariance=cov,
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        overlap_rho_prime=overlaps,
        h_t=dec.h,
        **sectors,
    )

    print(f"saved: {args.output}")
    print("leading eigenvalues:", eigvals[:5])
    print("leading overlaps:", overlaps[:5])
    print("closure_error:", sectors["closure_error"])


if __name__ == "__main__":
    main()
