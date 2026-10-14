"""Block-tridiagonal symmetric positive-definite linear algebra in JAX.

A symmetric matrix ``M`` of size ``(K·B) × (K·B)`` whose nonzero entries lie
within ``B − 1`` of the diagonal is block-tridiagonal with ``B × B`` blocks. It is
stored as two arrays:

* ``diag`` ``(K, B, B)``: the diagonal blocks ``M[k, k]``;
* ``sub`` ``(K − 1, B, B)``: the sub-diagonal blocks ``M[k + 1, k]``
  (the super-diagonal blocks are their transposes).

The Cholesky factor ``M = L Lᵀ`` is block lower-bidiagonal and is stored the same
way: ``Ld`` ``(K, B, B)`` holds the lower-triangular diagonal blocks ``L[k, k]`` and
``Ls`` ``(K − 1, B, B)`` the blocks ``L[k + 1, k]``.

Every function is a ``lax.scan`` over the K blocks, so it is jit-compatible,
differentiable, costs ``O(K·B³)``, and batches over leading axes with ``jax.vmap``.
The stellar template (:mod:`enuma.inference.stellar_gp`) uses this for its normal
equations, whose bandwidth is set by the LSF.
"""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

__all__ = [
    "block_cholesky",
    "block_solve",
    "block_logdet",
    "block_inverse_diag",
    "block_matvec",
    "block_to_dense",
    "jacobi_scale",
]


def _pad_front(sub: jnp.ndarray) -> jnp.ndarray:
    """``(K − 1, B, B)`` → ``(K, B, B)`` with a zero block in front, so block ``k``
    holds ``M[k, k − 1]`` (zero for ``k = 0``) and every scan step looks the same."""
    return jnp.concatenate([jnp.zeros_like(sub[:1]), sub], axis=0)


def _pad_back(sub: jnp.ndarray) -> jnp.ndarray:
    """``(K − 1, B, B)`` → ``(K, B, B)`` with a zero block at the end, so block ``k``
    holds ``L[k + 1, k]`` (zero for the last block)."""
    return jnp.concatenate([sub, jnp.zeros_like(sub[:1])], axis=0)


def block_cholesky(diag: jnp.ndarray, sub: jnp.ndarray):
    """Cholesky factor of a block-tridiagonal SPD matrix.

    Recurrence over blocks::

        L[k, k − 1] = M[k, k − 1] · L[k − 1, k − 1]⁻ᵀ
        L[k, k]     = chol( M[k, k] − L[k, k − 1] · L[k, k − 1]ᵀ )

    Returns ``(Ld, Ls)``: ``Ld`` ``(K, B, B)`` lower-triangular diagonal blocks and
    ``Ls`` ``(K − 1, B, B)`` the blocks ``L[k + 1, k]``.
    """
    B = diag.shape[-1]

    def step(Ld_prev, inputs):
        D_k, E_k = inputs
        Ls_k = solve_triangular(Ld_prev, E_k.T, lower=True).T
        Ld_k = jnp.linalg.cholesky(D_k - Ls_k @ Ls_k.T)
        return Ld_k, (Ld_k, Ls_k)

    _, (Ld, Ls_all) = jax.lax.scan(step, jnp.eye(B, dtype=diag.dtype),
                                   (diag, _pad_front(sub)))
    return Ld, Ls_all[1:]


def block_solve(Ld: jnp.ndarray, Ls: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    """Solve ``M x = rhs`` from the factor of :func:`block_cholesky`.

    ``rhs`` is ``(K, B)``; forward substitution ``L y = rhs`` then backward
    substitution ``Lᵀ x = y``, each one ``lax.scan``.
    """
    def forward(y_prev, inputs):
        Ld_k, Lprev_k, r_k = inputs
        y_k = solve_triangular(Ld_k, r_k - Lprev_k @ y_prev, lower=True)
        return y_k, y_k

    _, y = jax.lax.scan(forward, jnp.zeros_like(rhs[0]), (Ld, _pad_front(Ls), rhs))

    def backward(x_next, inputs):
        Ld_k, Lnext_k, y_k = inputs
        x_k = solve_triangular(Ld_k, y_k - Lnext_k.T @ x_next, lower=True, trans="T")
        return x_k, x_k

    _, x = jax.lax.scan(backward, jnp.zeros_like(rhs[0]), (Ld, _pad_back(Ls), y),
                        reverse=True)
    return x


def block_logdet(Ld: jnp.ndarray) -> jnp.ndarray:
    """``ln det M = 2 Σ ln diag(L)`` from the factor of :func:`block_cholesky`."""
    return 2.0 * jnp.sum(jnp.log(jnp.diagonal(Ld, axis1=-2, axis2=-1)))


def block_inverse_diag(Ld: jnp.ndarray, Ls: jnp.ndarray) -> jnp.ndarray:
    """Diagonal of ``M⁻¹``, ``(K, B)``, from the factor of :func:`block_cholesky`.

    Block Takahashi recursion, a reverse scan that only ever forms the
    block-tridiagonal part ``Σ`` of the inverse::

        Σ[k + 1, k] = −Σ[k + 1, k + 1] · L[k + 1, k] · L[k, k]⁻¹
        Σ[k, k]     = L[k, k]⁻ᵀ ( L[k, k]⁻¹ − L[k + 1, k]ᵀ · Σ[k + 1, k] )
    """
    B = Ld.shape[-1]
    eye = jnp.eye(B, dtype=Ld.dtype)

    def step(S_next, inputs):
        Ld_k, Lnext_k = inputs
        Linv = solve_triangular(Ld_k, eye, lower=True)
        S_off = -S_next @ Lnext_k @ Linv
        S_kk = Linv.T @ (Linv - Lnext_k.T @ S_off)
        return S_kk, jnp.diagonal(S_kk)

    _, d = jax.lax.scan(step, jnp.zeros_like(Ld[0]), (Ld, _pad_back(Ls)), reverse=True)
    return d


def block_matvec(diag: jnp.ndarray, sub: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """``M x`` for ``x`` of shape ``(K, B)``."""
    y = jnp.einsum("kij,kj->ki", diag, x)
    y = y.at[1:].add(jnp.einsum("kij,kj->ki", sub, x[:-1]))
    y = y.at[:-1].add(jnp.einsum("kji,kj->ki", sub, x[1:]))
    return y


def block_to_dense(diag: jnp.ndarray, sub: jnp.ndarray) -> jnp.ndarray:
    """Assemble the dense ``(K·B, K·B)`` matrix (for tests and small problems)."""
    K, B, _ = diag.shape
    M = jnp.zeros((K * B, K * B), dtype=diag.dtype)
    for k in range(K):
        M = M.at[k * B:(k + 1) * B, k * B:(k + 1) * B].set(diag[k])
    for k in range(K - 1):
        M = M.at[(k + 1) * B:(k + 2) * B, k * B:(k + 1) * B].set(sub[k])
        M = M.at[k * B:(k + 1) * B, (k + 1) * B:(k + 2) * B].set(sub[k].T)
    return M


def jacobi_scale(diag: jnp.ndarray, sub: jnp.ndarray):
    """Symmetric diagonal scaling ``M̃ = S M S``, ``S = diag(1/√M_jj)``, so ``M̃``
    has a unit diagonal (better conditioned in float32).

    Returns ``(diag̃, sub̃, s)`` with ``s`` ``(K, B)`` the diagonal of ``S``. Undo it
    with ``x = s · x̃`` for solves, ``ln det M = ln det M̃ − 2 Σ ln s``, and
    ``diag(M⁻¹) = s² · diag(M̃⁻¹)``.
    """
    s = 1.0 / jnp.sqrt(jnp.diagonal(diag, axis1=-2, axis2=-1))
    diag_s = s[:, :, None] * diag * s[:, None, :]
    sub_s = s[1:, :, None] * sub * s[:-1, None, :]
    return diag_s, sub_s, s
