from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from wvsr.mc import MCConfig, ExternalField, run_mc, save_run


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    p = argparse.ArgumentParser(description="Run NVT Lennard-Jones slab Metropolis MC")
    p.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p.add_argument("--output", type=Path, default=Path("results/raw/mc_stage1.npz"))
    p.add_argument("--field-amplitude", type=float, default=0.0)
    p.add_argument("--field-nx", type=int, default=0, help="integer Fourier index for optional parallel field")
    p.add_argument("--field-nz", type=int, default=0, help="integer Fourier index for optional normal field")
    p.add_argument("--observe-nx", type=int, nargs="*", default=None,
                   help="optional extra integer parallel Fourier indices to save in rho_q; merged with config n_x")
    p.add_argument("--seed", type=int, default=None, help="optional RNG seed override (useful for replicas)")
    args = p.parse_args()

    cfg_raw = load_config(args.config)
    s = cfg_raw["simulation"]
    init = cfg_raw["initialization"]
    wv = cfg_raw["wavevectors"]

    cfg = MCConfig(
        temperature=float(s["temperature"]),
        box=tuple(float(x) for x in s["box"]),
        n_particles=int(s["n_particles"]),
        cutoff=float(s["cutoff"]),
        neighbor_skin=float(s.get("neighbor_skin", 0.5)),
        max_displacement=float(s["max_displacement"]),
        equilibration_sweeps=int(s["equilibration_sweeps"]),
        production_sweeps=int(s["production_sweeps"]),
        sample_every=int(s["sample_every"]),
        z_bins=int(s["z_bins"]),
        seed=int(args.seed if args.seed is not None else s.get("seed", 0)),
    )

    lx, _, lz = cfg.box
    nx = np.asarray(wv["n_x"], dtype=int)
    extras = [] if args.observe_nx is None else list(args.observe_nx)
    if args.field_nx != 0:
        extras.append(int(args.field_nx))
    if extras:
        nx = np.unique(np.concatenate([nx, np.asarray(extras, dtype=int)]))
    qx = 2.0 * np.pi * nx / lx

    field = ExternalField(
        amplitude=float(args.field_amplitude),
        qx=2.0 * np.pi * int(args.field_nx) / lx,
        qz=2.0 * np.pi * int(args.field_nz) / lz,
    )

    result = run_mc(
        cfg,
        liquid_density=float(init["liquid_density"]),
        liquid_fraction_z=float(init["liquid_fraction_z"]),
        min_separation=float(init["min_separation"]),
        qx_values=qx,
        field=field,
    )
    save_run(args.output, result)

    print(f"saved: {args.output}")
    print(f"samples: {len(result['sample_sweeps'])}")
    print(f"acceptance_rate: {result['acceptance_rate']:.4f}")
    print(f"final_max_displacement: {result['final_max_displacement']:.5f}")
    print(f"seed: {cfg.seed}")
    print("nx observed:", nx)
    print("qx:", qx)


if __name__ == "__main__":
    main()
