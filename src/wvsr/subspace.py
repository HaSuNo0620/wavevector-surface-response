from __future__ import annotations

import numpy as np


def _quadrature_weights(z: np.ndarray | None, n: int) -> np.ndarray:
    if z is None:
        return np.ones(n, dtype=float)
    zz = np.asarray(z, dtype=float)
    if zz.ndim != 1 or len(zz) != n:
        raise ValueError("z must be one-dimensional with length matching vector dimension")
    if n < 2:
        raise ValueError("need at least two grid points")
    w = np.empty(n, dtype=float)
    dz = np.diff(zz)
    w[0] = dz[0] / 2.0
    w[-1] = dz[-1] / 2.0
    if n > 2:
        w[1:-1] = 0.5 * (dz[:-1] + dz[1:])
    return w


def _weighted_orthonormal_basis(columns: np.ndarray, z: np.ndarray | None = None, rtol: float = 1e-12) -> np.ndarray:
    """Return Euclidean-orthonormal basis after applying sqrt quadrature weights.

    Input shape is (n_grid, n_vectors). The returned basis lives in the
    weighted coordinate y = sqrt(W) x, so ordinary Euclidean dot products
    equal the intended z-inner-products.
    """
    a = np.asarray(columns, dtype=np.complex128)
    if a.ndim != 2:
        raise ValueError("columns must have shape (n_grid, n_vectors)")
    w = _quadrature_weights(z, a.shape[0])
    aw = np.sqrt(w)[:, None] * a
    if aw.shape[1] == 0:
        return np.zeros((aw.shape[0], 0), dtype=np.complex128)
    u, s, _ = np.linalg.svd(aw, full_matrices=False)
    if len(s) == 0 or s[0] == 0:
        return np.zeros((aw.shape[0], 0), dtype=np.complex128)
    rank = int(np.sum(s > rtol * s[0]))
    return u[:, :rank]


def principal_angle_cos2(
    eigenspace_vectors: np.ndarray,
    templates: np.ndarray,
    z: np.ndarray | None = None,
) -> np.ndarray:
    """Squared cosines of principal angles between two subspaces.

    Parameters
    ----------
    eigenspace_vectors:
        Array of shape (n_grid, m), usually the first m covariance eigenvectors.
    templates:
        Array of shape (n_templates, n_grid).

    Returns
    -------
    cos2:
        Descending squared singular values in [0, 1]. For the two-interface
        capillary space there are at most two values. A value near one means
        one capillary direction lies inside the chosen eigenspace.
    """
    e = np.asarray(eigenspace_vectors, dtype=np.complex128)
    t = np.asarray(templates, dtype=np.complex128)
    if e.ndim != 2 or t.ndim != 2 or e.shape[0] != t.shape[1]:
        raise ValueError("expected eigenspace (n_grid,m) and templates (k,n_grid)")
    qe = _weighted_orthonormal_basis(e, z)
    qt = _weighted_orthonormal_basis(t.T, z)
    if qe.shape[1] == 0 or qt.shape[1] == 0:
        return np.zeros(0, dtype=float)
    s = np.linalg.svd(qt.conj().T @ qe, compute_uv=False)
    return np.clip((s.real ** 2), 0.0, 1.0)


def cumulative_subspace_capture(
    eigenvectors: np.ndarray,
    templates: np.ndarray,
    z: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Measure how rapidly the leading covariance eigenspace captures a template subspace.

    For each m=1..M, E_m is the span of the first m eigenvectors. We compute
    the principal-angle cos^2 values between E_m and the template subspace.
    The scalar capture fraction is

        F_m = Tr(P_E_m P_template) / dim(template)

    which equals the mean cos^2 over all template directions, padding missing
    principal angles with zeros when m < dim(template).
    """
    v = np.asarray(eigenvectors, dtype=np.complex128)
    t = np.asarray(templates, dtype=np.complex128)
    if v.ndim != 2 or t.ndim != 2 or v.shape[0] != t.shape[1]:
        raise ValueError("dimension mismatch")

    qt = _weighted_orthonormal_basis(t.T, z)
    k = qt.shape[1]
    if k == 0:
        raise ValueError("template subspace has zero rank")

    capture = np.zeros(v.shape[1], dtype=float)
    cos2_first = np.zeros(v.shape[1], dtype=float)
    cos2_second = np.zeros(v.shape[1], dtype=float)

    for m in range(1, v.shape[1] + 1):
        cos2 = principal_angle_cos2(v[:, :m], t, z)
        padded = np.zeros(k, dtype=float)
        padded[: min(k, len(cos2))] = cos2[:k]
        capture[m - 1] = float(np.sum(padded) / k)
        if k >= 1:
            cos2_first[m - 1] = padded[0]
        if k >= 2:
            cos2_second[m - 1] = padded[1]

    return {
        "m": np.arange(1, v.shape[1] + 1, dtype=int),
        "capture_fraction": capture,
        "principal_cos2_1": cos2_first,
        "principal_cos2_2": cos2_second,
    }
