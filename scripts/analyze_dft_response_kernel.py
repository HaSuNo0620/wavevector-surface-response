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


def phase_from_name(path: Path) -> str:
    name = path.name
    for phase in ("slab", "liquid", "vapor"):
        if f"_{phase}_" in name or name.startswith(f"{phase}_"):
            return phase
    raise ValueError(f"cannot infer phase from {name}")


def replica_from_name(path: Path) -> int:
    stem = path.stem
    token = stem.split("rep")[-1]
    return int(token)


def integrated_modes(run: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    b = box(run)
    area = b[0] * b[1]
    dz = b[2] / len(run["z"])
    qx = np.asarray(run["qx"], dtype=float)
    rhoq = np.asarray(run["rho_q_tqz"])
    Z = area * dz * np.sum(rhoq, axis=2)
    return qx, Z


def profile(run: dict[str, np.ndarray]) -> np.ndarray:
    return np.mean(np.asarray(run["rho0_tz"], dtype=float), axis=0)


def structure_factor(run: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    qx, Z = integrated_modes(run)
    N = run["positions_final"].shape[0]
    Zc = Z - np.mean(Z, axis=0, keepdims=True)
    S = np.mean(np.abs(Zc) ** 2, axis=0) / float(N)
    return qx, S


def var_X(run: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    qx, Z = integrated_modes(run)
    X = Z.real
    Xc = X - np.mean(X, axis=0, keepdims=True)
    return qx, np.mean(Xc**2, axis=0)


def interp_c(k_samples: np.ndarray, c_samples: np.ndarray, k: np.ndarray) -> np.ndarray:
    out = np.interp(k, k_samples, c_samples, left=c_samples[0], right=0.0)
    out = np.where(k <= k_samples[-1], out, 0.0)
    return out


def periodic_cq_matrix(q: float, z: np.ndarray, Lz: float,
                       k_samples: np.ndarray, c_samples: np.ndarray) -> np.ndarray:
    nz = len(z)
    dz = Lz / nz
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)
    kr = np.sqrt(q*q + kz*kz)
    ck = interp_c(k_samples, c_samples, kr)
    cq_grid = np.fft.ifft(ck).real * nz / Lz
    idx = (np.arange(nz)[:, None] - np.arange(nz)[None, :]) % nz
    return cq_grid[idx]


def response_J(rho_z: np.ndarray, q: float, Lz: float,
               k_samples: np.ndarray, c_samples: np.ndarray,
               rho_floor: float = 1e-5) -> tuple[float, float, bool]:
    nz = len(rho_z)
    dz = Lz / nz
    z = (np.arange(nz) + 0.5) * dz
    C = periodic_cq_matrix(q, z, Lz, k_samples, c_samples)
    rho = np.maximum(np.asarray(rho_z, dtype=float), rho_floor)
    L = np.diag(1.0 / rho) - dz * C
    L = 0.5 * (L + L.T)
    eig = np.linalg.eigvalsh(L)
    min_eig = float(eig[0])
    stable = bool(min_eig > 1e-8)
    ones = np.ones(nz)
    if stable:
        sol = np.linalg.solve(L, ones)
    else:
        sol = np.linalg.pinv(L, rcond=1e-10) @ ones
    J = float(dz * ones @ sol)
    return J, min_eig, stable


def main() -> None:
    p = argparse.ArgumentParser(description="Frozen-c2 RY prediction for the wavevector-resolved surface free-energy response")
    p.add_argument("input_dir", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results/processed/dft_response"))
    p.add_argument("--n-interfaces", type=int, default=2)
    args = p.parse_args()

    runs: dict[str, list[tuple[int, dict[str, np.ndarray]]]] = {"slab": [], "liquid": [], "vapor": []}
    for path in sorted(args.input_dir.rglob("*.npz")):
        ph = phase_from_name(path)
        runs[ph].append((replica_from_name(path), load_run(path)))
    for ph in runs:
        if not runs[ph]:
            raise RuntimeError(f"missing {ph} baseline")
        runs[ph].sort(key=lambda x: x[0])

    slab0 = runs["slab"][0][1]
    liquid0 = runs["liquid"][0][1]
    vapor0 = runs["vapor"][0][1]
    bs, bl, bv = box(slab0), box(liquid0), box(vapor0)
    area = float(bs[0] * bs[1])
    Vs, Vl_ref, Vv_ref = map(float, (np.prod(bs), np.prod(bl), np.prod(bv)))
    Ns = slab0["positions_final"].shape[0]
    Nl = liquid0["positions_final"].shape[0]
    Nv = vapor0["positions_final"].shape[0]
    rho_bar, rho_l, rho_v = Ns / Vs, Nl / Vl_ref, Nv / Vv_ref
    f_l = float(np.clip((rho_bar - rho_v) / (rho_l - rho_v), 0.0, 1.0))
    wl = f_l * Vs / Vl_ref
    wv = (1.0 - f_l) * Vs / Vv_ref
    T = float(metadata(slab0)["temperature"])
    beta = 1.0 / T

    # Replica-averaged slab profile and bulk-liquid S(k).
    rho0_slab = np.mean([profile(r) for _, r in runs["slab"]], axis=0)
    q_list = []
    S_list = []
    for _, r in runs["liquid"]:
        q, S = structure_factor(r)
        q_list.append(q)
        S_list.append(S)
    q_samples = q_list[0]
    if not all(np.allclose(q_samples, q) for q in q_list[1:]):
        raise RuntimeError("liquid replicas have different q grids")
    S_rep = np.asarray(S_list)
    S_mean = np.mean(S_rep, axis=0)
    S_se = np.std(S_rep, axis=0, ddof=1) / np.sqrt(len(S_rep)) if len(S_rep) > 1 else np.zeros_like(S_mean)
    S_safe = np.maximum(S_mean, 1e-6)
    c_samples = (1.0 - 1.0 / S_safe) / rho_l

    # FDT response from the same A=0 replicas.
    var_rep: dict[str, np.ndarray] = {}
    for ph in ("slab", "liquid", "vapor"):
        vals = []
        for _, r in runs[ph]:
            q, v = var_X(r)
            if not np.allclose(q_samples, q):
                raise RuntimeError(f"{ph} q grid differs from liquid grid")
            vals.append(v)
        var_rep[ph] = np.asarray(vals)

    nrep = min(len(runs["slab"]), len(runs["liquid"]), len(runs["vapor"]))
    chi_fdt_rep = []
    for ir in range(nrep):
        vex = var_rep["slab"][ir] - wl * var_rep["liquid"][ir] - wv * var_rep["vapor"][ir]
        chi_fdt_rep.append(-beta * vex / (args.n_interfaces * area))
    chi_fdt_rep = np.asarray(chi_fdt_rep)
    chi_fdt = np.mean(chi_fdt_rep, axis=0)
    chi_fdt_se = np.std(chi_fdt_rep, axis=0, ddof=1) / np.sqrt(nrep) if nrep > 1 else np.zeros_like(chi_fdt)

    # Frozen-c2 RY response. The same liquid-reference c(k) is used in slab, liquid and vapor operators.
    rho_liq = np.full(len(liquid0["z"]), rho_l)
    rho_vap = np.full(len(vapor0["z"]), rho_v)
    rows = []
    for iq, q in enumerate(q_samples):
        Js, es, ss = response_J(rho0_slab, float(q), float(bs[2]), q_samples, c_samples)
        Jl, el, sl = response_J(rho_liq, float(q), float(bl[2]), q_samples, c_samples)
        Jv, ev, sv = response_J(rho_vap, float(q), float(bv[2]), q_samples, c_samples)
        Jex = Js - wl * Jl - wv * Jv
        chi_ry = -beta * Jex / (2.0 * args.n_interfaces)
        rows.append({
            "nx": int(round(q * bs[0] / (2.0*np.pi))),
            "qx": float(q),
            "S_liquid": float(S_mean[iq]),
            "se_S_liquid": float(S_se[iq]),
            "c_liquid": float(c_samples[iq]),
            "chi_ry": float(chi_ry),
            "chi_fdt": float(chi_fdt[iq]),
            "se_chi_fdt": float(chi_fdt_se[iq]),
            "min_eig_slab": es,
            "min_eig_liquid": el,
            "min_eig_vapor": ev,
            "stable_slab": ss,
            "stable_liquid": sl,
            "stable_vapor": sv,
        })

    df = pd.DataFrame(rows)
    w = 1.0 / np.maximum(df["se_chi_fdt"].to_numpy()**2, 1e-12)
    chi0 = float(np.sum(w * df["chi_fdt"]) / np.sum(w))
    df["chi_local_constant_fit"] = chi0
    df["resid_fdt_minus_constant"] = df["chi_fdt"] - chi0
    df["resid_fdt_minus_ry"] = df["chi_fdt"] - df["chi_ry"]

    rmse_const = float(np.sqrt(np.mean(df["resid_fdt_minus_constant"]**2)))
    rmse_ry = float(np.sqrt(np.mean(df["resid_fdt_minus_ry"]**2)))
    chi2_const = float(np.sum((df["resid_fdt_minus_constant"] / np.maximum(df["se_chi_fdt"], 1e-6))**2))
    chi2_ry = float(np.sum((df["resid_fdt_minus_ry"] / np.maximum(df["se_chi_fdt"], 1e-6))**2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "dft_response_prediction.csv", index=False)
    pd.DataFrame([{
        "temperature": T, "area": area, "n_interfaces": args.n_interfaces,
        "rho_bar": rho_bar, "rho_liquid": rho_l, "rho_vapor": rho_v,
        "liquid_volume_fraction": f_l, "bulk_liquid_weight": wl, "bulk_vapor_weight": wv,
        "n_replicas": nrep, "constant_fit_chi0": chi0,
        "rmse_constant": rmse_const, "rmse_ry": rmse_ry,
        "chi2_constant": chi2_const, "chi2_ry": chi2_ry,
        "model_note": "frozen-c2 RY using liquid S(k); c(k>kmax)=0",
    }]).to_csv(args.output_dir / "dft_response_summary.csv", index=False)

    plt.figure()
    plt.errorbar(df["qx"], df["S_liquid"], yerr=df["se_S_liquid"], marker="o", capsize=2)
    plt.xlabel(r"$q$")
    plt.ylabel(r"$S_l(q)$")
    plt.tight_layout()
    plt.savefig(args.output_dir / "bulk_structure_factor.png", dpi=180)
    plt.close()

    plt.figure()
    plt.errorbar(df["qx"], df["chi_fdt"], yerr=df["se_chi_fdt"], marker="o", capsize=2, label="A=0 FDT")
    plt.plot(df["qx"], df["chi_local_constant_fit"], linestyle="--", label="best constant (local null)")
    plt.plot(df["qx"], df["chi_ry"], marker="o", label="frozen-c2 RY")
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel(r"$q_\parallel$")
    plt.ylabel(r"$\chi_\gamma(q)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "chi_local_vs_nonlocal_dft.png", dpi=180)
    plt.close()

    plt.figure()
    plt.plot(df["qx"], df["min_eig_slab"], marker="o", label="slab")
    plt.plot(df["qx"], df["min_eig_liquid"], marker="o", label="liquid")
    plt.plot(df["qx"], df["min_eig_vapor"], marker="o", label="vapor")
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel(r"$q_\parallel$")
    plt.ylabel("minimum Hessian eigenvalue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "dft_hessian_stability.png", dpi=180)
    plt.close()

    print(df.to_string(index=False))
    print(f"constant chi0={chi0:.6g}  RMSE={rmse_const:.6g}  chi2={chi2_const:.6g}")
    print(f"frozen-c2 RY RMSE={rmse_ry:.6g}  chi2={chi2_ry:.6g}")


if __name__ == "__main__":
    main()
