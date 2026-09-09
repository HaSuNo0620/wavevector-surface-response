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


def mean_var(run: dict[str, np.ndarray], qx_target: float) -> tuple[float, float]:
    X = x_series(run, qx_target)
    mean = float(np.mean(X))
    var = float(np.mean((X - mean) ** 2))
    return mean, var


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def weighted_fit(M: np.ndarray, y: np.ndarray, se: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    se = np.asarray(se, dtype=float)
    good = np.isfinite(se) & (se > 0)
    if not np.all(good):
        finite = se[good]
        fallback = float(np.median(finite)) if finite.size else 1.0
        se = np.where(good, se, fallback)
    w = 1.0 / se**2
    A = M.T @ (w[:, None] * M)
    b = M.T @ (w * y)
    cov = np.linalg.pinv(A)
    coef = cov @ b
    resid = y - M @ coef
    chi2 = float(np.sum((resid / se) ** 2))
    return coef, cov, chi2


def main() -> None:
    p = argparse.ArgumentParser(description="Test q-independent interfacial susceptibility using strong +/-A scans")
    p.add_argument("input_dir", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results/processed/ti_constant_test"))
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
        meanX, varX = mean_var(run, qfield)
        nx = int(round(qfield * box(run)[0] / (2.0 * np.pi)))
        finite_rows.append({"phase": phase, "replica": rep, "A": A, "absA": abs(A),
                            "qx": qfield, "nx": nx, "mean_X": meanX, "var_X": varX})

    scan = pd.DataFrame(finite_rows)
    if scan.empty:
        raise RuntimeError("no finite-A runs found")
    reps = sorted(scan.replica.unique())
    for rep in reps:
        for phase in ("slab", "liquid", "vapor"):
            if (phase, rep) not in baselines:
                raise RuntimeError(f"missing baseline phase={phase}, replica={rep}")

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

    # Build the interfacial derivative separately for each replica. This preserves
    # the cross-phase combination before replica averaging.
    rep_rows = []
    keys = scan[["nx", "qx", "A", "absA", "replica"]].drop_duplicates()
    for _, key in keys.iterrows():
        q = float(key.qx); A = float(key.A); rep = int(key.replica); nx = int(key.nx)
        sub = scan[(scan.replica == rep) & np.isclose(scan.qx, q) & np.isclose(scan.A, A)]
        means = {}
        for phase in ("slab", "liquid", "vapor"):
            row = sub[sub.phase == phase]
            if len(row) != 1:
                raise RuntimeError(f"missing finite point phase={phase}, q={q}, A={A}, rep={rep}")
            means[phase] = float(row.iloc[0].mean_X)
        g = (means["slab"] - wl*means["liquid"] - wv*means["vapor"]) / (args.n_interfaces * area)
        rep_rows.append({"nx": nx, "qx": q, "A": A, "absA": abs(A), "replica": rep, "g": g})
    rep_df = pd.DataFrame(rep_rows)

    odd_rows = []
    for (nx, q, a, rep), g in rep_df.groupby(["nx", "qx", "absA", "replica"]):
        gp = g[np.isclose(g.A, a)]
        gm = g[np.isclose(g.A, -a)]
        if len(gp) != 1 or len(gm) != 1:
            raise RuntimeError(f"missing +/- pair q={q}, a={a}, rep={rep}")
        odd = 0.5 * (float(gp.iloc[0].g) - float(gm.iloc[0].g))
        even = 0.5 * (float(gp.iloc[0].g) + float(gm.iloc[0].g))
        odd_rows.append({"nx": int(nx), "qx": float(q), "absA": float(a), "replica": int(rep),
                         "g_odd": odd, "g_even": even, "ratio": odd/float(a)})
    odd_rep = pd.DataFrame(odd_rows)

    points = (odd_rep.groupby(["nx", "qx", "absA"], as_index=False)
              .agg(g_odd=("g_odd", "mean"), se_g_odd=("g_odd", sem),
                   g_even=("g_even", "mean"), se_g_even=("g_even", sem),
                   ratio=("ratio", "mean"), se_ratio=("ratio", sem), n_replicas=("replica", "size")))

    # For every q, extrapolate R(q,A)=g_odd/A linearly in A^2 to A->0.
    qfit_rows = []
    for q, sub in points.groupby("qx"):
        sub = sub.sort_values("absA")
        x = sub.absA.to_numpy(float)**2
        y = sub.ratio.to_numpy(float)
        se = sub.se_ratio.to_numpy(float)
        M = np.column_stack([np.ones_like(x), x])
        coef, cov, chi2 = weighted_fit(M, y, se)
        qfit_rows.append({"nx": int(sub.nx.iloc[0]), "qx": float(q),
                          "chi_extrap": float(coef[0]), "se_chi_extrap": float(np.sqrt(max(cov[0,0], 0.0))),
                          "c3": float(coef[1]), "se_c3": float(np.sqrt(max(cov[1,1], 0.0))),
                          "chi2_A2_fit": chi2, "dof_A2_fit": max(len(sub)-2, 0),
                          "max_abs_even": float(np.max(np.abs(sub.g_even)))})
    qfits = pd.DataFrame(qfit_rows).sort_values("qx")

    # Null H0: all q share one A->0 susceptibility.
    yq = qfits.chi_extrap.to_numpy(float)
    seq = qfits.se_chi_extrap.to_numpy(float)
    M0 = np.ones((len(qfits), 1))
    coef0, cov0, chi2_0 = weighted_fit(M0, yq, seq)
    chi0 = float(coef0[0]); se_chi0 = float(np.sqrt(max(cov0[0,0], 0.0)))

    # FDT from baseline for comparison.
    fdt_rows = []
    for q in qfits.qx:
        vals = []
        for rep in reps:
            _, vs = mean_var(baselines[("slab", rep)], q)
            _, vl = mean_var(baselines[("liquid", rep)], q)
            _, vv = mean_var(baselines[("vapor", rep)], q)
            vals.append(-beta * (vs - wl*vl - wv*vv) / (args.n_interfaces * area))
        fdt_rows.append({"qx": q, "chi_fdt": float(np.mean(vals)), "se_chi_fdt": sem(np.asarray(vals))})
    fdt = pd.DataFrame(fdt_rows)
    qfits = qfits.merge(fdt, on="qx", how="left")
    qfits["chi0_constant"] = chi0
    qfits["resid_from_constant"] = qfits.chi_extrap - chi0

    n_q = len(qfits)
    summary = pd.DataFrame([{
        "temperature": T, "area": area, "n_interfaces": args.n_interfaces,
        "rho_bar": rho_bar, "rho_liquid": rho_l, "rho_vapor": rho_v,
        "liquid_volume_fraction": f_l, "bulk_liquid_weight": wl, "bulk_vapor_weight": wv,
        "n_replicas": len(reps), "n_q": n_q,
        "constant_chi0": chi0, "se_constant_chi0": se_chi0,
        "chi2_constant_q": chi2_0, "dof_constant_q": max(n_q-1, 0),
        "rmse_q_about_constant": float(np.sqrt(np.mean((yq-chi0)**2))),
        "interpretation": "H0 is q-independent chi; q-specific intercepts come from R=g_odd/A extrapolated linearly in A^2"
    }])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rep_df.to_csv(args.output_dir / "finite_response_by_replica.csv", index=False)
    odd_rep.to_csv(args.output_dir / "odd_even_by_replica.csv", index=False)
    points.to_csv(args.output_dir / "ratio_points.csv", index=False)
    qfits.to_csv(args.output_dir / "q_extrapolated_chi.csv", index=False)
    summary.to_csv(args.output_dir / "constant_chi_summary.csv", index=False)

    plt.figure()
    for q, sub in points.groupby("qx"):
        plt.errorbar(sub.absA**2, sub.ratio, yerr=sub.se_ratio, marker="o", capsize=3, label=f"q={q:.2f}")
    plt.axhline(chi0, linestyle="--", label=f"constant H0={chi0:.3g}")
    plt.xlabel(r"$A^2$")
    plt.ylabel(r"$g_{\rm odd}(q,A)/A$")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(args.output_dir / "ratio_vs_A2.png", dpi=180)
    plt.close()

    plt.figure()
    plt.errorbar(qfits.qx, qfits.chi_extrap, yerr=qfits.se_chi_extrap, marker="o", capsize=3, label=r"TI $A\to0$")
    plt.axhline(chi0, linestyle="--", label="constant susceptibility H0")
    plt.errorbar(qfits.qx, qfits.chi_fdt, yerr=qfits.se_chi_fdt, marker="o", capsize=3, label="FDT")
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel(r"$q_\parallel$")
    plt.ylabel(r"$\chi_\gamma$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "chi_constant_test.png", dpi=180)
    plt.close()

    plt.figure()
    plt.errorbar(qfits.qx, qfits.resid_from_constant, yerr=qfits.se_chi_extrap, marker="o", capsize=3)
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel(r"$q_\parallel$")
    plt.ylabel(r"$\chi_\gamma(q)-\chi_0$")
    plt.tight_layout()
    plt.savefig(args.output_dir / "q_residual_from_constant.png", dpi=180)
    plt.close()

    print(summary.to_string(index=False))
    print(qfits.to_string(index=False))


if __name__ == "__main__":
    main()
