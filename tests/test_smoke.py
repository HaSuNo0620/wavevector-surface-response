import numpy as np

from wvsr.mc import MCConfig, density_modes, initialize_slab, run_mc
from wvsr.analysis import covariance_kernel, translational_template, decompose_mode, sector_variances


def test_density_mode_shapes():
    rng = np.random.default_rng(1)
    pos = initialize_slab(
        64,
        (6.0, 6.0, 12.0),
        liquid_density=0.65,
        liquid_fraction_z=0.60,
        min_separation=0.8,
        rng=rng,
    )
    z, rho0, rhoq = density_modes(pos, np.array([6.0, 6.0, 12.0]), 24, np.array([2*np.pi/6.0, 4*np.pi/6.0]))
    assert z.shape == (24,)
    assert rho0.shape == (24,)
    assert rhoq.shape == (2, 24)
    assert np.isfinite(rho0).all()
    assert np.isfinite(rhoq.real).all()


def test_short_mc_and_projection_closure():
    cfg = MCConfig(
        temperature=1.0,
        box=(6.0, 6.0, 12.0),
        n_particles=64,
        cutoff=2.5,
        neighbor_skin=0.5,
        max_displacement=0.08,
        equilibration_sweeps=2,
        production_sweeps=8,
        sample_every=1,
        z_bins=24,
        seed=3,
    )
    qx = np.array([2*np.pi/6.0])
    out = run_mc(
        cfg,
        liquid_density=0.65,
        liquid_fraction_z=0.60,
        min_separation=0.8,
        qx_values=qx,
    )
    rho = out["rho_q_tqz"][:, 0, :]
    rho0 = out["rho0_tz"].mean(axis=0)
    z = out["z"]
    cov = covariance_kernel(rho)
    assert cov.shape == (24, 24)
    phi = translational_template(rho0, z)
    dec = decompose_mode(rho, phi, z)
    dz = float(np.mean(np.diff(z)))
    centered = rho - rho.mean(axis=0, keepdims=True)
    total = centered.sum(axis=1) * dz
    h_obs = dec.rho_parallel.sum(axis=1) * dz
    p_obs = dec.rho_perp.sum(axis=1) * dz
    sec = sector_variances(total, h_obs, p_obs)
    assert abs(sec["closure_error"]) < 1e-10
    assert 0.0 < out["acceptance_rate"] < 1.0
