from __future__ import annotations

import argparse
from pathlib import Path
import csv

import numpy as np

from wvsr.analysis import (
    covariance_kernel,
    translational_template,
    two_interface_templates,
    diagonalize_covariance,
    template_overlaps,
    subspace_overlaps,
    decompose_subspace,
    sector_variances,
)
from wvsr.subspace import cumulative_subspace_capture


def analyze_one_q(z: np.ndarray, rho0: np.ndarray, rho: np.ndarray, n_modes: int) -> dict:
    cov = covariance_kernel(rho)
    phi_global = translational_template(rho0, z)
    phi_interfaces, centers = two_interface_templates(rho0, z)
    eigvals, eigvecs = diagonalize_covariance(cov, n_modes)

    global_overlaps = template_overlaps(eigvecs, phi_global, z)
    capillary_overlaps = subspace_overlaps(eigvecs, phi_interfaces, z)
    cumulative = cumulative_subspace_capture(eigvecs, phi_interfaces, z)

    dec = decompose_subspace(rho, phi_interfaces, z)
    dz = float(np.mean(np.diff(z)))
    centered = rho - rho.mean(axis=0, keepdims=True)
    total = centered.sum(axis=1) * dz
    cap_obs = dec.rho_parallel.sum(axis=1) * dz
    pack_obs = dec.rho_perp.sum(axis=1) * dz
    sectors = sector_variances(total, cap_obs, pack_obs)

    return {
        "covariance": cov,
        "global_template": phi_global,
        "interface_templates": phi_interfaces,
        "interface_centers": centers,
        "eigenvalues": eigvals,
        "eigenvectors": eigvecs,
        "global_overlaps": global_overlaps,
        "capillary_overlaps": capillary_overlaps,
        "cumulative": cumulative,
        "capillary_coefficients_t2": dec.coefficients,
        "sectors": sectors,
    }


def _value_at_m(arr: np.ndarray, m: int) -> float:
    idx = min(max(int(m), 1), len(arr)) - 1
    return float(arr[idx])


def _m_to_capture(capture: np.ndarray, threshold: float) -> int:
    hit = np.where(capture >= threshold)[0]
    return int(hit[0] + 1) if len(hit) else -1


def main() -> None:
    p = argparse.ArgumentParser(description="Goldstone/packing mode analysis of MC density modes")
    p.add_argument("input", type=Path, help="MC NPZ from scripts/run_mc.py")
    p.add_argument("--output-dir", type=Path, default=Path("results/processed/modes"))
    p.add_argument("--n-modes", type=int, default=16)
    args = p.parse_args()

    d = np.load(args.input)
    z = d["z"]
    qx = d["qx"]
    rho0_tz = d["rho0_tz"]
    rho0 = rho0_tz.mean(axis=0)
    rho_all = d["rho_q_tqz"]
    if rho_all.ndim != 3 or rho_all.shape[1] != len(qx):
        raise ValueError("expected rho_q_tqz with shape (time, q, z)")

    half = len(rho0_tz) // 2
    if half >= 2:
        profile_drift_rms = float(np.sqrt(np.mean((rho0_tz[:half].mean(axis=0) - rho0_tz[half:].mean(axis=0)) ** 2)))
    else:
        profile_drift_rms = float("nan")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for iq, q in enumerate(qx):
        rho = rho_all[:, iq, :]
        out = analyze_one_q(z, rho0, rho, args.n_modes)
        sectors = out["sectors"]
        dominant = int(np.argmax(out["capillary_overlaps"]))
        second_order = np.argsort(out["capillary_overlaps"])[::-1]
        second = int(second_order[1]) if len(second_order) > 1 else dominant
        cumulative = out["cumulative"]
        capture = cumulative["capture_fraction"]

        np.savez_compressed(
            args.output_dir / f"q_{iq:03d}.npz",
            qx=q,
            z=z,
            rho0=rho0,
            global_template=out["global_template"],
            interface_templates=out["interface_templates"],
            interface_centers=out["interface_centers"],
            covariance=out["covariance"],
            eigenvalues=out["eigenvalues"],
            eigenvectors=out["eigenvectors"],
            overlap_global_translation=out["global_overlaps"],
            overlap_capillary_subspace=out["capillary_overlaps"],
            cumulative_m=cumulative["m"],
            cumulative_capture=capture,
            principal_cos2_1=cumulative["principal_cos2_1"],
            principal_cos2_2=cumulative["principal_cos2_2"],
            capillary_coefficients_t2=out["capillary_coefficients_t2"],
            profile_drift_rms=profile_drift_rms,
            **sectors,
        )
        rows.append({
            "iq": iq,
            "qx": float(q),
            "dominant_capillary_mode": dominant,
            "capillary_overlap": float(out["capillary_overlaps"][dominant]),
            "capillary_eigenvalue": float(out["eigenvalues"][dominant]),
            "second_capillary_mode": second,
            "second_capillary_overlap": float(out["capillary_overlaps"][second]),
            "second_capillary_eigenvalue": float(out["eigenvalues"][second]),
            "global_translation_overlap_of_dominant": float(out["global_overlaps"][dominant]),
            "largest_eigenvalue": float(out["eigenvalues"][0]),
            "capture_m2": _value_at_m(capture, 2),
            "capture_m4": _value_at_m(capture, 4),
            "capture_m8": _value_at_m(capture, 8),
            "capture_m16": _value_at_m(capture, 16),
            "principal1_m4": _value_at_m(cumulative["principal_cos2_1"], 4),
            "principal2_m4": _value_at_m(cumulative["principal_cos2_2"], 4),
            "principal1_m8": _value_at_m(cumulative["principal_cos2_1"], 8),
            "principal2_m8": _value_at_m(cumulative["principal_cos2_2"], 8),
            "m_capture_50": _m_to_capture(capture, 0.50),
            "m_capture_80": _m_to_capture(capture, 0.80),
            "interface_z_1": float(out["interface_centers"][0]),
            "interface_z_2": float(out["interface_centers"][1]),
            "profile_drift_rms": profile_drift_rms,
            **sectors,
        })

    csv_path = args.output_dir / "mode_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {csv_path}")
    print(f"profile_drift_rms={profile_drift_rms:.6g}")
    for row in rows:
        print(
            f"q={row['qx']:.6f} capture4={row['capture_m4']:.4f} "
            f"capture8={row['capture_m8']:.4f} cap_overlap={row['capillary_overlap']:.4f} "
            f"closure={row['closure_error']:.3e}"
        )


if __name__ == "__main__":
    main()
