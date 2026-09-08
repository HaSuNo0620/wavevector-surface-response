from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ModeDecomposition:
    h: np.ndarray
    rho_parallel: np.ndarray
    rho_perp: np.ndarray
    template: np.ndarray


def covariance_kernel(rho_q_tz: np.ndarray, *, center: bool = True) -> np.ndarray:
    """Return Hermitian covariance C(z,z') from complex rho_q(z,t).

    Parameters
    ----------
    rho_q_tz:
        Complex array with shape (n_samples, n_z).
    center:
        Subtract the time mean at each z before forming covariance.
    """
    x = np.asarray(rho_q_tz, dtype=np.complex128)
    if x.ndim != 2:
        raise ValueError("rho_q_tz must have shape (n_samples, n_z)")
    if center:
        x = x - x.mean(axis=0, keepdims=True)
    n = x.shape[0]
    if n < 2:
        raise ValueError("need at least two samples")
    c = (x.conj().T @ x) / (n - 1)
    return 0.5 * (c + c.conj().T)


def translational_template(rho0_z: np.ndarray, z: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    """Construct the Goldstone/displacement template phi_h(z)=-d rho0/dz."""
    rho0 = np.asarray(rho0_z, dtype=float)
    zz = np.asarray(z, dtype=float)
    if rho0.shape != zz.shape:
        raise ValueError("rho0_z and z must have identical shapes")
    phi = -np.gradient(rho0, zz, edge_order=2)
    if normalize:
        norm = np.sqrt(np.trapz(np.abs(phi) ** 2, zz))
        if norm == 0:
            raise ValueError("zero translational template norm")
        phi = phi / norm
    return phi.astype(np.complex128)


def diagonalize_covariance(cov: np.ndarray, n_modes: int | None = None):
    """Diagonalize a Hermitian covariance and return descending eigenpairs."""
    c = np.asarray(cov, dtype=np.complex128)
    vals, vecs = np.linalg.eigh(c)
    order = np.argsort(vals.real)[::-1]
    vals = vals[order].real
    vecs = vecs[:, order]
    if n_modes is not None:
        vals = vals[:n_modes]
        vecs = vecs[:, :n_modes]
    return vals, vecs


def template_overlaps(eigenvectors: np.ndarray, template: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
    """Return normalized squared overlap of each eigenvector with template."""
    v = np.asarray(eigenvectors, dtype=np.complex128)
    phi = np.asarray(template, dtype=np.complex128)
    if v.shape[0] != phi.size:
        raise ValueError("eigenvectors and template dimensions differ")

    if z is None:
        inner = lambda a, b: np.vdot(a, b)
    else:
        zz = np.asarray(z, dtype=float)
        if zz.size != phi.size:
            raise ValueError("z dimension differs from template")
        inner = lambda a, b: np.trapz(np.conj(a) * b, zz)

    pnorm = np.real(inner(phi, phi))
    out = []
    for i in range(v.shape[1]):
        vi = v[:, i]
        vnorm = np.real(inner(vi, vi))
        num = abs(inner(vi, phi)) ** 2
        out.append(float(num / (vnorm * pnorm)))
    return np.asarray(out)


def decompose_mode(rho_q_tz: np.ndarray, template: np.ndarray, z: np.ndarray | None = None) -> ModeDecomposition:
    """Project each density-mode snapshot onto phi_h and its orthogonal complement.

    delta rho_q(z,t) = h_q(t) phi_h(z) + delta rho_q^perp(z,t)
    """
    x = np.asarray(rho_q_tz, dtype=np.complex128)
    x = x - x.mean(axis=0, keepdims=True)
    phi = np.asarray(template, dtype=np.complex128)
    if x.shape[1] != phi.size:
        raise ValueError("rho_q_tz and template dimensions differ")

    if z is None:
        denom = np.vdot(phi, phi)
        h = (x @ np.conj(phi)) / denom
    else:
        zz = np.asarray(z, dtype=float)
        denom = np.trapz(np.conj(phi) * phi, zz)
        h = np.array([np.trapz(np.conj(phi) * row, zz) / denom for row in x])

    parallel = h[:, None] * phi[None, :]
    perp = x - parallel
    return ModeDecomposition(h=h, rho_parallel=parallel, rho_perp=perp, template=phi)


def projected_observable(rho_tz: np.ndarray, form_factor_z: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
    """Project rho_q(z,t) onto an external-field z form factor."""
    x = np.asarray(rho_tz, dtype=np.complex128)
    f = np.asarray(form_factor_z, dtype=np.complex128)
    if x.shape[1] != f.size:
        raise ValueError("dimensions differ")
    if z is None:
        return x @ np.conj(f)
    zz = np.asarray(z, dtype=float)
    return np.array([np.trapz(row * np.conj(f), zz) for row in x])


def sector_variances(total_x: np.ndarray, h_x: np.ndarray, perp_x: np.ndarray) -> dict[str, float]:
    """Variance/covariance decomposition for X = X_h + X_perp.

    Returns quantities satisfying var_total = var_hh + 2*cross + var_perpperp
    up to floating-point/statistical consistency.
    """
    xt = np.asarray(total_x, dtype=np.complex128)
    xh = np.asarray(h_x, dtype=np.complex128)
    xp = np.asarray(perp_x, dtype=np.complex128)

    def centered(a):
        return a - a.mean()

    t, h, p = map(centered, (xt, xh, xp))
    var_total = float(np.mean(np.abs(t) ** 2).real)
    var_hh = float(np.mean(np.abs(h) ** 2).real)
    var_pp = float(np.mean(np.abs(p) ** 2).real)
    cross = float(np.mean(np.real(h * np.conj(p))))
    closure = var_total - (var_hh + 2.0 * cross + var_pp)
    return {
        "var_total": var_total,
        "var_hh": var_hh,
        "cross_h_perp": cross,
        "var_perpperp": var_pp,
        "closure_error": float(closure),
    }
