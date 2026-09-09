from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_run(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def metadata(run: dict[str, np.ndarray]) -> dict:
    m = run["metadata"]
    m = m.item() if getattr(m, "shape", ()) == () else m
    return json.loads(m) if isinstance(m, str) else dict(m)


def box(run: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(metadata(run)["box"], dtype=float)


def mean_and_var_X(run: dict[str, np.ndarray], qx_target: float) -> tuple[float, float, float]:
    b = box(run)
    area = b[0] * b[1]
    dz = b[2] / len(run["z"])
    qx = np.asarray(run["qx"], dtype=float)
    iq = int(np.argmin(np.abs(qx - qx_target)))
    if not np.isclose(qx[iq], qx_target, rtol=0.0, atol=1e-10):
        raise ValueError(f"target q={qx_target} not sampled; closest={qx[iq]}")
    rhoq = np.asarray(run["rho_q_tqz"])
    Z = area * dz * np.sum(rhoq[:, iq, :], axis=1)
    X = Z.real
    mean = float(np.mean(X))
    var = float(np.mean((X - mean) ** 2))
    se = float(np.std(X, ddof=1) / np.sqrt(len(X)))
    return mean, var, se


def phase_from_name(path: Path) -> str:
    name = path.name
    for phase in ("slab", "liquid", "vapor"):
        if f"_{phase}_" in name or name.startswith(f"{phase}_") or f"baseline_{phase}" in name:
            return phase
    raise ValueError(f"cannot infer phase from {name}")


def gather(input_dir: Path) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    rows = []
    baselines: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(input_dir.rglob("*.npz")):
        run = load_run(path)
        m = metadata(run)
        phase = phase_from_name(path)
        A = float(m["field"]["amplitude"])
        qfield = float(m["field"]["qx"])
        if abs(A) < 1e-15:
            baselines[phase] = run
            continue
        meanX, varX, seX = mean_and_var_X(run, qfield)
        nx = int(round(qfield * box(run)[0] / (2.0 * np.pi)))
        rows.append({"phase": phase, "A": A, "qx": qfield, "nx": nx,
                     "mean_X": meanX, "var_X": varX, "se_X": seX,
                     "volume": float(np.prod(box(run)))})
    return pd.DataFrame(rows), baselines


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze finite-A TI scan and compare with A=0 FDT")
    p.add_argument("input_dir", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results/processed/ti_scan"))
    p.add_argument("--n-interfaces", type=int, default=2)
    args = p.parse_args()

    scan, base = gather(args.input_dir)
    for phase in ("slab", "liquid", "vapor"):
        if phase not in base:
            raise RuntimeError(f"missing A=0 baseline for {phase}")

    bs, bl, bv = box(base["slab"]), box(base["liquid"]), box(base["vapor"])
    Vs, Vl_ref, Vv_ref = map(float, (np.prod(bs), np.prod(bl), np.prod(bv)))
    area = float(bs[0] * bs[1])
    Ns = int(base["slab"]["positions_final"].shape[0])
    Nl = int(base["liquid"]["positions_final"].shape[0])
    Nv = int(base["vapor"]["positions_final"].shape[0])
    rho_bar, rho_l, rho_v = Ns / Vs, Nl / Vl_ref, Nv / Vv_ref
    f_l = float(np.clip((rho_bar - rho_v) / (rho_l - rho_v), 0.0, 1.0))
    V_l, V_v = f_l * Vs, (1.0 - f_l) * Vs
    wl, wv = V_l / Vl_ref, V_v / Vv_ref
    T = float(metadata(base["slab"])["temperature"])
    beta = 1.0 / T

    qvals = sorted(scan["qx"].unique())
    point_rows = []
    fit_rows = []

    for q in qvals:
        sub = scan[np.isclose(scan["qx"], q)].copy()
        merged = None
        for phase in ("slab", "liquid", "vapor"):
            s = sub[sub.phase == phase][["A", "mean_X", "se_X"]].rename(
                columns={"mean_X": f"mean_{phase}", "se_X": f"se_{phase}"})
            merged = s if merged is None else merged.merge(s, on="A", how="inner")
        if merged is None or len(merged) < 2:
            continue
        merged = merged.sort_values("A")
        A = merged["A"].to_numpy(float)
        Xex = merged["mean_slab"].to_numpy() - wl * merged["mean_liquid"].to_numpy() - wv * merged["mean_vapor"].to_numpy()
        y = Xex / (args.n_interfaces * area)  # d gamma / dA
        se_y = np.sqrt(merged["se_slab"].to_numpy()**2 + (wl*merged["se_liquid"].to_numpy())**2 + (wv*merged["se_vapor"].to_numpy())**2) / (args.n_interfaces * area)

        # Add exact symmetry point y(0)=0 for commensurate q>0.
        Aint = np.concatenate([[0.0], A])
        yint = np.concatenate([[0.0], y])
        dgamma = np.zeros_like(Aint)
        for i in range(1, len(Aint)):
            dgamma[i] = dgamma[i-1] + 0.5 * (yint[i] + yint[i-1]) * (Aint[i] - Aint[i-1])

        # Odd derivative fit: dgamma/dA = chi A + c3 A^3.
        M = np.column_stack([A, A**3])
        chi_deriv, c3 = np.linalg.lstsq(M, y, rcond=None)[0]

        # Integrated fit: Delta gamma = 1/2 chi A^2 + c4 A^4.
        Mint = np.column_stack([0.5 * A**2, A**4])
        chi_int, c4 = np.linalg.lstsq(Mint, dgamma[1:], rcond=None)[0]

        # FDT from A=0 baselines for the same q.
        _, var_s, _ = mean_and_var_X(base["slab"], q)
        _, var_l, _ = mean_and_var_X(base["liquid"], q)
        _, var_v, _ = mean_and_var_X(base["vapor"], q)
        var_ex = var_s - wl * var_l - wv * var_v
        chi_fdt = -beta * var_ex / (args.n_interfaces * area)

        nx = int(round(q * bs[0] / (2.0*np.pi)))
        fit_rows.append({"nx": nx, "qx": q, "chi_ti_derivative": chi_deriv,
                         "chi_ti_integral": chi_int, "chi_fdt": chi_fdt,
                         "c3_derivative": c3, "c4_integral": c4,
                         "var_excess_fdt": var_ex})
        for ai, yi, sei, dgi in zip(A, y, se_y, dgamma[1:]):
            point_rows.append({"nx": nx, "qx": q, "A": ai, "dgamma_dA": yi,
                               "se_dgamma_dA": sei, "delta_gamma_ti": dgi})

    points = pd.DataFrame(point_rows)
    fits = pd.DataFrame(fit_rows).sort_values("qx")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points.to_csv(args.output_dir / "ti_scan_points.csv", index=False)
    fits.to_csv(args.output_dir / "ti_chi_comparison.csv", index=False)
    pd.DataFrame([{"temperature": T, "area": area, "n_interfaces": args.n_interfaces,
                   "rho_bar": rho_bar, "rho_liquid": rho_l, "rho_vapor": rho_v,
                   "liquid_volume_fraction": f_l, "bulk_liquid_weight": wl,
                   "bulk_vapor_weight": wv}]).to_csv(args.output_dir / "ti_metadata.csv", index=False)

    plt.figure()
    for q, g in points.groupby("qx"):
        plt.errorbar(g["A"], g["dgamma_dA"], yerr=g["se_dgamma_dA"], marker="o", label=f"q={q:.2f}")
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("A")
    plt.ylabel(r"$d\gamma/dA$")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.output_dir / "dgamma_dA_vs_A.png", dpi=180)
    plt.close()

    plt.figure()
    for q, g in points.groupby("qx"):
        plt.plot(g["A"]**2, g["delta_gamma_ti"], marker="o", label=f"q={q:.2f}")
    plt.xlabel(r"$A^2$")
    plt.ylabel(r"$\Delta\gamma(A,q)$")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.output_dir / "delta_gamma_vs_A2.png", dpi=180)
    plt.close()

    plt.figure()
    plt.plot(fits["qx"], fits["chi_ti_derivative"], marker="o", label="finite-A slope")
    plt.plot(fits["qx"], fits["chi_ti_integral"], marker="o", label="finite-A TI")
    plt.plot(fits["qx"], fits["chi_fdt"], marker="o", label="A=0 FDT")
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel(r"$q_\parallel$")
    plt.ylabel(r"$\chi_\gamma(q)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "chi_ti_vs_fdt.png", dpi=180)
    plt.close()

    print(fits.to_string(index=False))


if __name__ == "__main__":
    main()
