from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    q = np.asarray([float(r["qx"]) for r in rows])
    vp = np.asarray([float(r["var_h_plus"]) for r in rows])
    vm = np.asarray([float(r["var_h_minus"]) for r in rows])
    return q, vp, vm


def aicc(n: int, k: int, rss: float) -> float:
    if rss <= 0:
        return -np.inf
    val = n * np.log(rss / n) + 2 * k
    if n > k + 1:
        val += 2 * k * (k + 1) / (n - k - 1)
    return float(val)


def fit_one(q: np.ndarray, y: np.ndarray, nfit: int) -> dict[str, float]:
    q = np.asarray(q[:nfit], dtype=float)
    y = np.asarray(y[:nfit], dtype=float)
    x = q * q

    # Constant model: finite variance with no resolved q dependence.
    c = float(np.mean(y))
    pred_const = np.full_like(y, c)
    rss_const = float(np.sum((y - pred_const) ** 2))

    # Pure massless capillary model: B/q^2.
    basis = 1.0 / x
    b = float(np.dot(basis, y) / np.dot(basis, basis))
    pred_massless = b / x
    rss_massless = float(np.sum((y - pred_massless) ** 2))

    # Massive capillary model in identifiable form y0/(1 + xi2 q^2).
    def massive(qv, y0, xi2):
        return y0 / (1.0 + xi2 * qv * qv)

    p0 = [max(float(y[0]), 1e-12), 0.1]
    popt, _ = curve_fit(
        massive,
        q,
        y,
        p0=p0,
        bounds=([0.0, 0.0], [np.inf, np.inf]),
        maxfev=20000,
    )
    y0, xi2 = map(float, popt)
    pred_massive = massive(q, y0, xi2)
    rss_massive = float(np.sum((y - pred_massive) ** 2))

    # Inverse-variance diagnostic: 1/y = alpha + beta q^2.
    beta, alpha = np.polyfit(x, 1.0 / y, 1)

    return {
        "nfit": int(nfit),
        "const_c": c,
        "rss_const": rss_const,
        "aicc_const": aicc(nfit, 1, rss_const),
        "massless_B": b,
        "rss_massless": rss_massless,
        "aicc_massless": aicc(nfit, 1, rss_massless),
        "massive_y0": y0,
        "massive_xi2": xi2,
        "massive_xi": float(np.sqrt(xi2)),
        "rss_massive": rss_massive,
        "aicc_massive": aicc(nfit, 2, rss_massive),
        "inverse_alpha": float(alpha),
        "inverse_beta": float(beta),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Compare massless, massive and constant capillary-coordinate models")
    p.add_argument("input", type=Path, help="capillary_coordinate_summary.csv")
    p.add_argument("--output-dir", type=Path, default=Path("results/processed/capillary_fit"))
    p.add_argument("--nfit", type=int, default=4, help="number of lowest-q points used for the primary comparison")
    args = p.parse_args()

    q, vp, vm = read_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, y in [("plus", vp), ("minus", vm)]:
        result = fit_one(q, y, args.nfit)
        rows.append({"mode": name, **result})

    out_csv = args.output_dir / "massive_model_comparison.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    qfit = q[:args.nfit]
    qplot = np.linspace(qfit.min(), qfit.max(), 300)
    for name, y, result in [("plus", vp, rows[0]), ("minus", vm, rows[1])]:
        fig, ax = plt.subplots()
        ax.plot(qfit, y[:args.nfit], marker="o", linestyle="none", label="data")
        ax.plot(qplot, np.full_like(qplot, result["const_c"]), label="constant")
        ax.plot(qplot, result["massless_B"] / (qplot * qplot), label=r"$B/q^2$")
        ax.plot(
            qplot,
            result["massive_y0"] / (1.0 + result["massive_xi2"] * qplot * qplot),
            label=r"$y_0/(1+\xi^2 q^2)$",
        )
        ax.set_xlabel(r"$q_\parallel$")
        ax.set_ylabel(rf"$\langle |h_{{{name[0]}}}|^2\rangle$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{name}_model_comparison.png", dpi=180)
        plt.close(fig)

    print(f"saved: {out_csv}")
    for row in rows:
        print(
            f"{row['mode']}: AICc const={row['aicc_const']:.4f} "
            f"massless={row['aicc_massless']:.4f} massive={row['aicc_massive']:.4f} "
            f"xi2={row['massive_xi2']:.6g} inverse_beta={row['inverse_beta']:.6g}"
        )


if __name__ == "__main__":
    main()
