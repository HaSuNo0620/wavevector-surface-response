from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_run(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def box_from_metadata(run: dict[str, np.ndarray]) -> np.ndarray:
    meta = run.get("metadata")
    if meta is None:
        raise KeyError("metadata missing from run")
    m = meta.item() if getattr(meta, "shape", ()) == () else meta
    if isinstance(m, str):
        import json
        m = json.loads(m)
    return np.asarray(m["box"], dtype=float)


def temperature_from_metadata(run: dict[str, np.ndarray]) -> float:
    meta = run["metadata"]
    m = meta.item() if getattr(meta, "shape", ()) == () else meta
    if isinstance(m, str):
        import json
        m = json.loads(m)
    return float(m["temperature"])


def x_variance(run: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    box = box_from_metadata(run)
    lx, ly, lz = box
    area = lx * ly
    z = np.asarray(run["z"], dtype=float)
    dz = lz / len(z)
    rhoq = np.asarray(run["rho_q_tqz"])
    # Z_q = sum_j exp(i q x_j) = A_xy int dz rho_q(z)
    Z = area * dz * np.sum(rhoq, axis=2)
    X = Z.real
    X = X - np.mean(X, axis=0, keepdims=True)
    var = np.mean(X * X, axis=0)
    return np.asarray(run["qx"], dtype=float), var


def main() -> None:
    p = argparse.ArgumentParser(description="Compute pilot bulk-subtracted interfacial susceptibility")
    p.add_argument("slab", type=Path)
    p.add_argument("bulk_liquid", type=Path)
    p.add_argument("bulk_vapor", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results/processed/interfacial_susceptibility"))
    p.add_argument("--n-interfaces", type=int, default=2)
    args = p.parse_args()

    slab = load_run(args.slab)
    liq = load_run(args.bulk_liquid)
    vap = load_run(args.bulk_vapor)

    qs, var_s = x_variance(slab)
    ql, var_l = x_variance(liq)
    qv, var_v = x_variance(vap)
    if not (np.allclose(qs, ql) and np.allclose(qs, qv)):
        raise ValueError("qx grids differ between slab and bulk references")

    bs = box_from_metadata(slab)
    bl = box_from_metadata(liq)
    bv = box_from_metadata(vap)
    Vs = float(np.prod(bs)); Vl_ref = float(np.prod(bl)); Vv_ref = float(np.prod(bv))
    As = float(bs[0] * bs[1])

    Ns = int(np.asarray(slab["positions_final"]).shape[0])
    Nl = int(np.asarray(liq["positions_final"]).shape[0])
    Nv = int(np.asarray(vap["positions_final"]).shape[0])
    rho_bar = Ns / Vs
    rho_l = Nl / Vl_ref
    rho_v = Nv / Vv_ref
    f_l = (rho_bar - rho_v) / (rho_l - rho_v)
    f_l = float(np.clip(f_l, 0.0, 1.0))
    V_l = f_l * Vs
    V_v = Vs - V_l

    var_density_l = var_l / Vl_ref
    var_density_v = var_v / Vv_ref
    var_bulk_match = V_l * var_density_l + V_v * var_density_v
    var_excess = var_s - var_bulk_match

    T = temperature_from_metadata(slab)
    beta = 1.0 / T
    chi_gamma = -beta * var_excess / (args.n_interfaces * As)

    out = pd.DataFrame({
        "qx": qs,
        "var_X_slab": var_s,
        "var_X_bulk_liquid_ref": var_l,
        "var_X_bulk_vapor_ref": var_v,
        "var_X_bulk_matched": var_bulk_match,
        "var_X_excess": var_excess,
        "chi_gamma_pilot": chi_gamma,
    })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_dir / "interfacial_susceptibility_summary.csv", index=False)

    meta = pd.DataFrame([{
        "temperature": T,
        "beta": beta,
        "area": As,
        "n_interfaces": args.n_interfaces,
        "rho_slab_mean": rho_bar,
        "rho_bulk_liquid": rho_l,
        "rho_bulk_vapor": rho_v,
        "liquid_volume_fraction": f_l,
        "liquid_volume": V_l,
        "vapor_volume": V_v,
        "note": "pilot bulk subtraction using phase densities matched to the current slab pilot; not publication coexistence data",
    }])
    meta.to_csv(args.output_dir / "interfacial_susceptibility_metadata.csv", index=False)

    plt.figure()
    plt.plot(qs, chi_gamma, marker="o")
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel(r"$q_\\parallel$")
    plt.ylabel(r"$\\chi_\\gamma(q)$ (pilot)")
    plt.tight_layout()
    plt.savefig(args.output_dir / "chi_gamma_vs_q.png", dpi=180)
    plt.close()

    plt.figure()
    plt.plot(qs, var_s, marker="o", label="slab")
    plt.plot(qs, var_bulk_match, marker="o", label="matched bulk")
    plt.plot(qs, var_excess, marker="o", label="excess")
    plt.xlabel(r"$q_\\parallel$")
    plt.ylabel(r"$\\mathrm{Var}(X_q)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "variance_bulk_subtraction_vs_q.png", dpi=180)
    plt.close()

    print(out.to_string(index=False))
    print(meta.to_string(index=False))


if __name__ == "__main__":
    main()
