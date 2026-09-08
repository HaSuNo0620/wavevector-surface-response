from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class MCConfig:
    temperature: float
    box: tuple[float, float, float]
    n_particles: int
    cutoff: float
    max_displacement: float
    equilibration_sweeps: int
    production_sweeps: int
    sample_every: int
    z_bins: int
    seed: int = 0
    neighbor_skin: float = 0.5
    epsilon: float = 1.0
    sigma: float = 1.0

    @property
    def beta(self) -> float:
        return 1.0 / self.temperature


@dataclass
class ExternalField:
    amplitude: float = 0.0
    qx: float = 0.0
    qz: float = 0.0
    phase: float = 0.0

    def energy(self, r: np.ndarray) -> float:
        return float(self.amplitude * math.cos(self.qx * r[0] + self.qz * r[2] + self.phase))


class VerletNeighbors:
    """Periodic Verlet neighbor list built with scipy cKDTree.

    The list contains pairs within cutoff+skin at the last rebuild. It remains
    valid while every particle has moved less than skin/2 since that rebuild.
    Trial moves that would cross the skin/2 bound must therefore trigger a
    rebuild *before* old/new local energies are evaluated.
    """

    def __init__(self, positions: np.ndarray, box: np.ndarray, cutoff: float, skin: float):
        self.box = np.asarray(box, dtype=float)
        self.cutoff = float(cutoff)
        self.skin = float(skin)
        self.radius = self.cutoff + self.skin
        self.reference = np.array(positions, dtype=float, copy=True)
        self.displacement = np.zeros_like(self.reference)
        self.neighbors: list[np.ndarray] = []
        self.rebuild(positions)

    def rebuild(self, positions: np.ndarray) -> None:
        pos = np.mod(np.asarray(positions, dtype=float), self.box)
        tree = cKDTree(pos, boxsize=self.box)
        pairs = tree.query_pairs(self.radius, output_type="ndarray")
        lists: list[list[int]] = [[] for _ in range(len(pos))]
        for i, j in pairs:
            lists[int(i)].append(int(j))
            lists[int(j)].append(int(i))
        self.neighbors = [np.asarray(v, dtype=np.int32) for v in lists]
        self.reference = pos.copy()
        self.displacement.fill(0.0)

    def trial_requires_rebuild(self, i: int, dr: np.ndarray) -> bool:
        candidate = self.displacement[i] + np.asarray(dr, dtype=float)
        return bool(np.linalg.norm(candidate) >= 0.5 * self.skin)

    def record_move(self, i: int, dr: np.ndarray) -> None:
        self.displacement[i] += dr


def minimum_image(dr: np.ndarray, box: np.ndarray) -> np.ndarray:
    return dr - box * np.rint(dr / box)


def lj_shifted_pair(r2: np.ndarray, cutoff: float, epsilon: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """Shifted Lennard-Jones pair energy; values at/above cutoff are zero."""
    r2 = np.asarray(r2, dtype=float)
    out = np.zeros_like(r2)
    mask = (r2 > 0.0) & (r2 < cutoff * cutoff)
    if not np.any(mask):
        return out
    sr2 = (sigma * sigma) / r2[mask]
    sr6 = sr2 ** 3
    sr12 = sr6 * sr6
    rc2 = (sigma / cutoff) ** 2
    rc6 = rc2 ** 3
    rc12 = rc6 * rc6
    shift = 4.0 * epsilon * (rc12 - rc6)
    out[mask] = 4.0 * epsilon * (sr12 - sr6) - shift
    return out


def local_energy(i: int, trial: np.ndarray, positions: np.ndarray, nbrs: VerletNeighbors, cfg: MCConfig, field: ExternalField) -> float:
    idx = nbrs.neighbors[i]
    if idx.size:
        dr = trial[None, :] - positions[idx]
        dr = minimum_image(dr, np.asarray(cfg.box))
        r2 = np.sum(dr * dr, axis=1)
        e_pair = float(np.sum(lj_shifted_pair(r2, cfg.cutoff, cfg.epsilon, cfg.sigma)))
    else:
        e_pair = 0.0
    return e_pair + field.energy(trial)


def initialize_slab(
    n_particles: int,
    box: tuple[float, float, float],
    liquid_density: float,
    liquid_fraction_z: float,
    min_separation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a dense central slab on a slightly jittered simple-cubic grid."""
    lx, ly, lz = map(float, box)
    slab_thickness = lz * float(liquid_fraction_z)
    target_volume = n_particles / float(liquid_density)
    if target_volume > lx * ly * slab_thickness * 1.20:
        raise ValueError(
            "requested N is too large for liquid_density/liquid_fraction_z; "
            "increase slab fraction or lower N"
        )

    spacing = max(float(min_separation), (1.0 / liquid_density) ** (1.0 / 3.0))
    nx = max(1, int(lx / spacing))
    ny = max(1, int(ly / spacing))
    nz = max(1, int(slab_thickness / spacing))
    while nx * ny * nz < n_particles:
        spacing *= 0.98
        nx = max(1, int(lx / spacing))
        ny = max(1, int(ly / spacing))
        nz = max(1, int(slab_thickness / spacing))
        if spacing < 0.75 * min_separation:
            raise ValueError("could not construct a nonoverlapping slab initializer")

    xs = (np.arange(nx) + 0.5) * lx / nx
    ys = (np.arange(ny) + 0.5) * ly / ny
    z0 = 0.5 * (lz - slab_thickness)
    zs = z0 + (np.arange(nz) + 0.5) * slab_thickness / nz
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    rng.shuffle(grid)
    pos = grid[:n_particles].copy()
    jitter = 0.05 * min(lx / nx, ly / ny, slab_thickness / nz)
    pos += rng.uniform(-jitter, jitter, size=pos.shape)
    return np.mod(pos, np.asarray(box, dtype=float))


def density_modes(positions: np.ndarray, box: np.ndarray, z_bins: int, qx_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return z centers, rho0(z), and rho_q(z) for all supplied qx values."""
    lx, ly, lz = box
    dz = lz / int(z_bins)
    z = (np.arange(z_bins) + 0.5) * dz
    bins = np.floor(np.mod(positions[:, 2], lz) / dz).astype(int)
    bins = np.clip(bins, 0, z_bins - 1)
    vol_bin = lx * ly * dz
    counts = np.bincount(bins, minlength=z_bins)
    rho0 = counts.astype(float) / vol_bin

    qx_values = np.asarray(qx_values, dtype=float)
    modes = np.zeros((len(qx_values), z_bins), dtype=np.complex128)
    for iq, qx in enumerate(qx_values):
        phase = np.exp(1j * qx * positions[:, 0])
        real = np.bincount(bins, weights=phase.real, minlength=z_bins)
        imag = np.bincount(bins, weights=phase.imag, minlength=z_bins)
        modes[iq] = (real + 1j * imag) / vol_bin
    return z, rho0, modes


def run_mc(
    cfg: MCConfig,
    *,
    liquid_density: float,
    liquid_fraction_z: float,
    min_separation: float,
    qx_values: np.ndarray,
    field: ExternalField | None = None,
    initial_positions: np.ndarray | None = None,
    adapt_displacement: bool = True,
) -> dict[str, np.ndarray | float | int | dict]:
    """Run NVT Metropolis MC and return sampled density modes."""
    rng = np.random.default_rng(cfg.seed)
    box = np.asarray(cfg.box, dtype=float)
    field = field or ExternalField()

    # A one-step trial must itself remain inside skin/2 after any pre-trial rebuild.
    safe_component_disp = 0.49 * cfg.neighbor_skin / math.sqrt(3.0)
    max_disp = min(float(cfg.max_displacement), safe_component_disp)

    positions = (
        np.array(initial_positions, dtype=float, copy=True)
        if initial_positions is not None
        else initialize_slab(
            cfg.n_particles,
            cfg.box,
            liquid_density,
            liquid_fraction_z,
            min_separation,
            rng,
        )
    )
    positions %= box
    if len(positions) != cfg.n_particles:
        raise ValueError("initial_positions particle count does not match config")

    nbrs = VerletNeighbors(positions, box, cfg.cutoff, cfg.neighbor_skin)
    accepted = 0
    attempted = 0
    samples_rho0: list[np.ndarray] = []
    samples_rhoq: list[np.ndarray] = []
    sample_sweeps: list[int] = []
    qx_values = np.asarray(qx_values, dtype=float)

    total_sweeps = cfg.equilibration_sweeps + cfg.production_sweeps
    window_attempts = 0
    window_accepts = 0

    for sweep in range(total_sweeps):
        order = rng.permutation(cfg.n_particles)
        for i in order:
            i = int(i)
            old = positions[i].copy()
            proposal_dr = rng.uniform(-max_disp, max_disp, size=3)
            trial = np.mod(old + proposal_dr, box)
            actual_dr = minimum_image(trial - old, box)

            # Critical correctness condition: if this *trial* would make the
            # cumulative displacement reach skin/2, rebuild at the current
            # configuration before evaluating both old and new energies.
            if nbrs.trial_requires_rebuild(i, actual_dr):
                nbrs.rebuild(positions)

            e_old = local_energy(i, old, positions, nbrs, cfg, field)
            e_new = local_energy(i, trial, positions, nbrs, cfg, field)
            dE = e_new - e_old
            attempted += 1
            window_attempts += 1
            if dE <= 0.0 or rng.random() < math.exp(-cfg.beta * dE):
                positions[i] = trial
                accepted += 1
                window_accepts += 1
                nbrs.record_move(i, actual_dr)

        if adapt_displacement and sweep < cfg.equilibration_sweeps and (sweep + 1) % 100 == 0:
            rate = window_accepts / max(window_attempts, 1)
            if rate < 0.30:
                max_disp *= 0.90
            elif rate > 0.50:
                max_disp *= 1.10
            max_disp = min(max_disp, safe_component_disp, 0.20 * min(box))
            window_attempts = 0
            window_accepts = 0

        if sweep >= cfg.equilibration_sweeps:
            prod_index = sweep - cfg.equilibration_sweeps
            if prod_index % cfg.sample_every == 0:
                _, rho0, rhoq = density_modes(positions, box, cfg.z_bins, qx_values)
                samples_rho0.append(rho0)
                samples_rhoq.append(rhoq)
                sample_sweeps.append(sweep)

    z, _, _ = density_modes(positions, box, cfg.z_bins, qx_values)
    return {
        "positions_final": positions,
        "z": z,
        "qx": qx_values,
        "rho0_tz": np.asarray(samples_rho0, dtype=float),
        "rho_q_tqz": np.asarray(samples_rhoq, dtype=np.complex128),
        "sample_sweeps": np.asarray(sample_sweeps, dtype=int),
        "acceptance_rate": accepted / max(attempted, 1),
        "final_max_displacement": max_disp,
        "metadata": {
            "temperature": cfg.temperature,
            "box": list(map(float, cfg.box)),
            "n_particles": cfg.n_particles,
            "cutoff": cfg.cutoff,
            "neighbor_skin": cfg.neighbor_skin,
            "epsilon": cfg.epsilon,
            "sigma": cfg.sigma,
            "field": {
                "amplitude": field.amplitude,
                "qx": field.qx,
                "qz": field.qz,
                "phase": field.phase,
            },
        },
    }


def save_run(path: str | Path, result: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(result["metadata"], sort_keys=True)
    np.savez_compressed(
        path,
        positions_final=result["positions_final"],
        z=result["z"],
        qx=result["qx"],
        rho0_tz=result["rho0_tz"],
        rho_q_tqz=result["rho_q_tqz"],
        sample_sweeps=result["sample_sweeps"],
        acceptance_rate=np.asarray(result["acceptance_rate"]),
        final_max_displacement=np.asarray(result["final_max_displacement"]),
        metadata=np.asarray(metadata),
    )
