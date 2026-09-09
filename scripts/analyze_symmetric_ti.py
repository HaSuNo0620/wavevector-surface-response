from __future__ import annotations

import argparse
import json
import re
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


def replica_from_name(path: Path) -> int:
    m = re.search(r"rep(\d+)", path.name)
    if not m:
        raise ValueError(f"replica index missing in {path.name}")
    return int(m.group(1))


def phase_from_name(path: Path) -> str:
    name = path.name
    for phase in ("slab", "liquid", "vapor"):
        if f"_{phase}_" in name or name.startswith(f"{phase}_") or f"baseline_{phase}" in name:
            return phase
    raise ValueError(f"cannot infer phase from {name}")


def x_series(run: dict[str, np.ndarray], qx_target: float) -> np.ndarray:
    b = box(run)
    area = b[0] * b[1]
    dz = b[2] / len(run["z"])
    qx = np.asarray(run["qx"], dtype=float)
    iq = int(np.argmin(np.abs(qx - qx_target)))
    if not np.isclose(qx[iq], qx_target, rtol=0.0, atol=1e-10):
        raise ValueError(f"target q={qx_target} not sampled; closest={qx[iq]}")
    rhoq = np.asarray(run["rho_q_tqz"])
    Z = area * dz * np.sum(rhoq[:, iq, :], axis=1)
    return Z.real


def mean_var_se(run: dict[str, np.ndarray], qx_target: float) -> tuple[float, float, float]:
    X = x_series(run, qx_target)
    mean = float(np.mean(X))
    var = float(np.mean((X - mean) ** 2))
    se = float(np.std(X, ddof=1) / np.sqrt(len(X)))
    return mean, var, se


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def weighted_lstsq(M: np.ndarray, y: np.ndarray, se: np.ndarray) -> np.ndarray:
    se = np.asarray(se, dtype=float)
    finite = np.isfinite(se) & (se > 0)
    if finite.all():
        w = 1.0 / se
        return np.linalg.lstsq(M * w[:, None], y * w, rcond=None)[0]
    return np.linalg.lstsq(M, y, rcond=None)[0]


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze +/-A replica-averaged TI using odd/even symmetry")
    p.add_argument("input_dir", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results/processed/ti_symmetric"))
    p.add_argument("--n-interfaces", type=int, default=2)
    args = p.parse_args()

    finite_rows = []
    baselines: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for path in sorted(args.input_dir.rglob("*.npz")):
        run = load_run(path)
        m = metadata(run)
        phase = phase_from_name(path)
        rep = replica_from_name(path)
        A = float(m["field"]["amplitude"])
        qfield = float(m["field"]["qx"])
        if abs(A) < 1e-15:
            baselines[(phase, rep)] = run
            continue
        meanX, varX, seX = mean_var_se(run, qfield)
        nx = int(round(qfield * box(run)[0] / (2.0 * np.pi)))
        finite_rows.append({"phase": phase, "replica": rep, "A": A, "absA": abs(A),
                            "qx": qfield, "nx": nx, "mean_X": meanX,
                            "var_X": varX, "se_X_internal": seX})

    scan = pd.DataFrame(finite_rows)
    if scan.empty:
        raise RuntimeError("no finite-A runs found")

    reps = sorted(scan["replica"].unique())
    for rep in reps:
        for phase in ("slab", "liquid", "vapor"):
            if (phase, rep) not in baselines:
                raise RuntimeError(f"missing baseline for phase={phase}, replica={rep}")

    ref = baselines[("slab", reps[0])]
    bs = box(ref)
    bl = box(baselines[("liquid", reps[0])])
    bv = box(baselines[("vapor", reps[0])])
    Vs, Vl_ref, Vv_ref = map(float, (np.prod(bs), np.prod(bl), np.prod(bv)))
    area = float(bs[0] * bs[1])
    Ns = int(ref["positions_final"].shape[0])
    Nl = int(baselines[("liquid", reps[0])]["positions_final"].shape[0])
    Nv = int(baselines[("vapor", reps[0])]["positions_final"].shape[0])
    rho_bar, rho_l, rho_v = Ns / Vs, Nl / Vl_ref, Nv / Vv_ref
    f_l = float(np.clip((rho_bar - rho_v) / (rho_l - rho_v), 0.0, 1.0))
    wl = f_l * Vs / Vl_ref
    wv = (1.0 - f_l) * Vs / Vv_ref
    T = float(metadata(ref)["temperature"])
    beta = 1.0 / T

    phase_avg = (scan.groupby(["phase", "nx", "qx", "A", "absA"], as_index=False)
                 .agg(mean_X=("mean_X", "mean"),
                      se_replica=("mean_X", sem),
                      n_replica=("mean_X", "size"),
                      mean_internal_se=("se_X_internal", "mean")))

    sym_rows = []
    fit_rows = []
    qvals = sorted(scan["qx"].unique())
    for q in qvals:
        qsub = phase_avg[np.isclose(phase_avg["qx"], q)]
        a_vals = sorted(qsub["absA"].unique())
        nx = int(round(q * bs[0] / (2.0 * np.pi)))
        q_sym = []

        for a in a_vals:
            vals = {}
            for sign, A in (("plus", a), ("minus", -a)):
                ps = qsub[np.isclose(qsub["A"], A)]
                means = {}
                ses = {}
                for phase in ("slab", "liquid", "vapor"):
                    row = ps[ps.phase == phase]
                    if len(row) != 1:
                        raise RuntimeError(f"missing/duplicate point q={q}, A={A}, phase={phase}")
                    means[phase] = float(row.iloc[0]["mean_X"])
                    ses[phase] = float(row.iloc[0]["se_replica"])
                y = (means["slab"] - wl*means["liquid"] - wv*means["vapor"]) / (args.n_interfaces * area)
                se_y = np.sqrt(ses["slab"]**2 + (wl*ses["liquid"])**2 + (wv*ses["vapor"])**2) / (args.n_interfaces * area)
                vals[sign] = (y, se_y)

            yp, sep = vals["plus"]
            ym, sem_ = vals["minus"]
            yodd = 0.5 * (yp - ym)
            yeven = 0.5 * (yp + ym)
            se_pair = 0.5 * np.sqrt(sep**2 + sem_**2)
            row = {"nx": nx, "qx": q, "absA": a, "y_plus": yp, "y_minus": ym,
                   "se_plus": sep, "se_minus": sem_, "y_odd": yodd,
                   "y_even": yeven, "se_pair": se_pair}
            sym_rows.append(row)
            q_sym.append(row)

        qdf = pd.DataFrame(q_sym).sort_values("absA")
        a = qdf["absA"].to_numpy(float)
        yodd = qdf["y_odd"].to_numpy(float)
        seodd = qdf["se_pair"].to_numpy(float)

        M = np.column_stack([a, a**3])
        chi_odd, c3 = weighted_lstsq(M, yodd, seodd)
        Mlin = a[:, None]
        chi_linear = float(weighted_lstsq(Mlin, yodd, seodd)[0])

        aint = np.concatenate([[0.0], a])
        yint = np.concatenate([[0.0], yodd])
        dgamma = np.zeros_like(aint)
        for i in range(1, len(aint)):
            dgamma[i] = dgamma[i-1] + 0.5*(yint[i] + yint[i-1])*(aint[i]-aint[i-1])
        Mint = np.column_stack([0.5*a**2, a**4])
        chi_int, c4 = np.linalg.lstsq(Mint, dgamma[1:], rcond=None)[0]

        fdt_rep = []
        for rep in reps:
            _, vs, _ = mean_var_se(baselines[("slab", rep)], q)
            _, vl, _ = mean_var_se(baselines[("liquid", rep)], q)
            _, vv, _ = mean_var_se(baselines[("vapor", rep)], q)
            vex = vs - wl*vl - wv*vv
            fdt_rep.append(-beta * vex / (args.n_interfaces * area))
        chi_fdt = float(np.mean(fdt_rep))
        se_fdt = sem(np.asarray(fdt_rep))

        fit_rows.append({"nx": nx, "qx": q, "chi_ti_odd_cubic": float(chi_odd),
                         "chi_ti_odd_linear": chi_linear, "chi_ti_integral": float(chi_int),
                         "chi_fdt": chi_fdt, "se_chi_fdt_replica": se_fdt,
                         "c3_odd": float(c3), "c4_integral": float(c4),
                         "max_abs_even": float(np.max(np.abs(qdf["y_even"]))),
                         "n_replicas": len(reps)})

    sym = pd.DataFrame(sym_rows).sort_values(["qx", "absA"])
    fits = pd.DataFrame(fit_rows).sort_values("qx")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scan.to_csv(args.output_dir / "replica_points_raw.csv", index=False)
    phase_avg.to_csv(args.output_dir / "phase_replica_averages.csv", index=False)
    sym.to_csv(args.output_dir / "symmetric_response_points.csv", index=False)
    fits.to_csv(args.output_dir / "symmetric_chi_comparison.csv", index=False)
    pd.DataFrame([{"temperature": T, "area": area, "n_interfaces": args.n_interfaces,
                   "rho_bar": rho_bar, "rho_liquid": rho_l, "rho_vapor": rho_v,
                   "liquid_volume_fraction": f_l, "bulk_liquid_weight": wl,
                   "bulk_vapor_weight": wv, "n_replicas": len(reps)}]).to_csv(
                       args.output_dir / "symmetric_ti_metadata.csv", index=False)

    plt.figure()
    for q, g in sym.groupby("qx"):
        plt.errorbar(g["absA"], g["y_odd"], yerr=g["se_pair"], marker="o", capsize=3,
                     label=f"q={q:.2f}")
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel(r"$|A|$")
    plt.ylabel(r"$[g(A)-g(-A)]/2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "odd_response_vs_absA.png", dpi=180)
    plt.close()

    plt.figure()
    for q, g in sym.groupby("qx"):
        plt.errorbar(g["absA"], g["y_even"], yerr=g["se_pair"], marker="o", capsize=3,
                     label=f"q={q:.2f}")
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel(r"$|A|$")
    plt.ylabel(r"$[g(A)+g(-A)]/2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "even_contamination_vs_absA.png", dpi=180)
    plt.close()

    plt.figure()
    plt.plot(fits["qx"], fits["chi_ti_odd_cubic"], marker="o", label="TI odd, cubic")
    plt.plot(fits["qx"], fits["chi_ti_odd_linear"], marker="o", label="TI odd, linear")
    plt.plot(fits["qx"], fits["chi_ti_integral"], marker="o", label="TI integrated")
    plt.errorbar(fits["qx"], fits["chi_fdt"], yerr=fits["se_chi_fdt_replica"],
                 marker="o", capsize=3, label="FDT replica mean")
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel(r"$q_\parallel$")
    plt.ylabel(r"$\chi_\gamma(q)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "chi_symmetric_ti_vs_fdt.png", dpi=180)
    plt.close()

    print(fits.to_string(index=False))


if __name__ == "__main__":
    main()
