from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ModeDecomposition:
    h: np.ndarray
    rho_parallel: np.ndarray
    rho_perp: np.ndarray
    template: np.ndarray


@dataclass
class SubspaceDecomposition:
    coefficients: np.ndarray
    rho_parallel: np.ndarray
    rho_perp: np.ndarray
    templates: np.ndarray


def covariance_kernel(rho_q_tz: np.ndarray, *, center: bool = True) -> np.ndarray:
    """Return Hermitian covariance C(z,z') from complex rho_q(z,t)."""
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


def _inner(a: np.ndarray, b: np.ndarray, z: np.ndarray | None = None) -> complex:
    if z is None:
        return np.vdot(a, b)
    zz = np.asarray(z, dtype=float)
    return np.trapezoid(np.conj(a) * b, zz)


def translational_template(rho0_z: np.ndarray, z: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    """Construct the whole-slab translation template phi(z)=-d rho0/dz.

    For a periodic liquid slab there are two interfaces. This whole-slab
    template is useful as a reference, but it should not be the only capillary
    basis because the two interfaces can fluctuate independently.
    """
    rho0 = np.asarray(rho0_z, dtype=float)
    zz = np.asarray(z, dtype=float)
    if rho0.shape != zz.shape:
        raise ValueError("rho0_z and z must have identical shapes")
    phi = -np.gradient(rho0, zz, edge_order=2)
    if normalize:
        norm = np.sqrt(np.real(_inner(phi, phi, zz)))
        if norm == 0:
            raise ValueError("zero translational template norm")
        phi = phi / norm
    return phi.astype(np.complex128)


def _smooth_periodic(y: np.ndarray, passes: int = 2) -> np.ndarray:
    out = np.asarray(y, dtype=float).copy()
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0
    offsets = np.arange(-2, 3)
    for _ in range(passes):
        acc = np.zeros_like(out)
        for w, off in zip(kernel, offsets):
            acc += w * np.roll(out, int(off))
        out = acc
    return out


def two_interface_templates(
    rho0_z: np.ndarray,
    z: np.ndarray,
    *,
    window_sigma: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two normalized local displacement templates for a periodic slab.

    The two interface centers are detected from the strongest separated peaks
    of |d rho0/dz| after mild periodic smoothing. Each local template is
    -rho0'(z) multiplied by a periodic Gaussian window around one interface.

    Returns
    -------
    templates:
        Complex array with shape (2, n_z).
    centers:
        Detected interface z positions, sorted in ascending order.
    """
    rho0 = np.asarray(rho0_z, dtype=float)
    zz = np.asarray(z, dtype=float)
    if rho0.shape != zz.shape or rho0.ndim != 1:
        raise ValueError("rho0_z and z must be one-dimensional with identical shapes")
    if len(zz) < 16:
        raise ValueError("need more z bins to identify two interfaces")

    smooth = _smooth_periodic(rho0)
    deriv = -np.gradient(smooth, zz, edge_order=2)
    strength = np.abs(deriv)

    first = int(np.argmax(strength))
    n = len(zz)
    periodic_index_distance = np.minimum(np.arange(n) - first, first - np.arange(n))
    periodic_index_distance = np.minimum(np.abs(np.arange(n) - first), n - np.abs(np.arange(n) - first))
    mask = periodic_index_distance >= max(4, n // 4)
    if not np.any(mask):
        raise ValueError("could not separate two interfaces")
    candidate = np.where(mask, strength, -np.inf)
    second = int(np.argmax(candidate))
    centers = np.sort(np.asarray([zz[first], zz[second]], dtype=float))

    dz = float(np.mean(np.diff(zz)))
    lz = dz * n
    separation = min(abs(centers[1] - centers[0]), lz - abs(centers[1] - centers[0]))
    sigma = float(window_sigma) if window_sigma is not None else max(2.0 * dz, separation / 6.0)

    templates = []
    for center in centers:
        d = np.abs(zz - center)
        d = np.minimum(d, lz - d)
        window = np.exp(-0.5 * (d / sigma) ** 2)
        phi = deriv * window
        norm = np.sqrt(np.real(_inner(phi, phi, zz)))
        if norm <= 0:
            raise ValueError("zero local interface-template norm")
        templates.append((phi / norm).astype(np.complex128))
    return np.asarray(templates), centers


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
    """Return normalized squared overlap of each eigenvector with one template."""
    v = np.asarray(eigenvectors, dtype=np.complex128)
    phi = np.asarray(template, dtype=np.complex128)
    if v.shape[0] != phi.size:
        raise ValueError("eigenvectors and template dimensions differ")

    pnorm = np.real(_inner(phi, phi, z))
    out = []
    for i in range(v.shape[1]):
        vi = v[:, i]
        vnorm = np.real(_inner(vi, vi, z))
        num = abs(_inner(vi, phi, z)) ** 2
        out.append(float(num / (vnorm * pnorm)))
    return np.asarray(out)


def subspace_overlaps(eigenvectors: np.ndarray, templates: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
    """Return squared overlap with the span of multiple templates.

    This is basis-invariant inside the template subspace, so a nearly
    degenerate pair of two-interface capillary modes is not spuriously lost
    when the covariance eigenvectors rotate within that pair.
    """
    v = np.asarray(eigenvectors, dtype=np.complex128)
    phi = np.asarray(templates, dtype=np.complex128)
    if phi.ndim != 2 or phi.shape[1] != v.shape[0]:
        raise ValueError("templates must have shape (n_templates, n_z)")

    g = np.asarray([[_inner(a, b, z) for b in phi] for a in phi], dtype=np.complex128)
    ginv = np.linalg.pinv(g, rcond=1e-12)
    out = []
    for i in range(v.shape[1]):
        vi = v[:, i]
        b = np.asarray([_inner(p, vi, z) for p in phi], dtype=np.complex128)
        proj_norm = np.real(np.conj(b) @ ginv @ b)
        vnorm = np.real(_inner(vi, vi, z))
        out.append(float(np.clip(proj_norm / vnorm, 0.0, 1.0)))
    return np.asarray(out)


def decompose_mode(rho_q_tz: np.ndarray, template: np.ndarray, z: np.ndarray | None = None) -> ModeDecomposition:
    """Project each density-mode snapshot onto one template and its complement."""
    x = np.asarray(rho_q_tz, dtype=np.complex128)
    x = x - x.mean(axis=0, keepdims=True)
    phi = np.asarray(template, dtype=np.complex128)
    if x.shape[1] != phi.size:
        raise ValueError("rho_q_tz and template dimensions differ")

    denom = _inner(phi, phi, z)
    h = np.asarray([_inner(phi, row, z) / denom for row in x])
    parallel = h[:, None] * phi[None, :]
    perp = x - parallel
    return ModeDecomposition(h=h, rho_parallel=parallel, rho_perp=perp, template=phi)


def decompose_subspace(rho_q_tz: np.ndarray, templates: np.ndarray, z: np.ndarray | None = None) -> SubspaceDecomposition:
    """Project density fluctuations onto a multi-template capillary subspace."""
    x = np.asarray(rho_q_tz, dtype=np.complex128)
    x = x - x.mean(axis=0, keepdims=True)
    phi = np.asarray(templates, dtype=np.complex128)
    if phi.ndim != 2 or phi.shape[1] != x.shape[1]:
        raise ValueError("templates must have shape (n_templates, n_z)")

    g = np.asarray([[_inner(a, b, z) for b in phi] for a in phi], dtype=np.complex128)
    ginv = np.linalg.pinv(g, rcond=1e-12)
    coeffs = []
    parallel = []
    for row in x:
        b = np.asarray([_inner(p, row, z) for p in phi], dtype=np.complex128)
        c = ginv @ b
        coeffs.append(c)
        parallel.append(np.sum(c[:, None] * phi, axis=0))
    coeffs = np.asarray(coeffs)
    parallel = np.asarray(parallel)
    perp = x - parallel
    return SubspaceDecomposition(coefficients=coeffs, rho_parallel=parallel, rho_perp=perp, templates=phi)


def projected_observable(rho_tz: np.ndarray, form_factor_z: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
    """Project rho_q(z,t) onto an external-field z form factor."""
    x = np.asarray(rho_tz, dtype=np.complex128)
    f = np.asarray(form_factor_z, dtype=np.complex128)
    if x.shape[1] != f.size:
        raise ValueError("dimensions differ")
    return np.asarray([_inner(np.conj(f), row, z) for row in x])


def sector_variances(total_x: np.ndarray, h_x: np.ndarray, perp_x: np.ndarray) -> dict[str, float]:
    """Variance/covariance decomposition for X = X_h + X_perp."""
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
