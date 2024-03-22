import numpy as np

from typing import Optional, Tuple, Union
import functools

from tinygrad import Tensor

"""Copied directly from e3nn-jax to avoid JAX imports"""


def _su2_cg(idx1, idx2, idx3):
    """Calculates the Clebsch-Gordon coefficient.

    Copied from the ``clebsch`` function in ``qutip``.
    """
    from fractions import Fraction
    import math

    j1, m1 = idx1
    j2, m2 = idx2
    j3, m3 = idx3

    if m3 != m1 + m2:
        return 0
    vmin = int(max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    def f(n):
        assert n == round(n)
        return math.factorial(round(n))

    C = (
        (2.0 * j3 + 1.0)
        * Fraction(
            f(j3 + j1 - j2)
            * f(j3 - j1 + j2)
            * f(j1 + j2 - j3)
            * f(j3 + m3)
            * f(j3 - m3),
            f(j1 + j2 + j3 + 1) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2),
        )
    ) ** 0.5

    S = 0
    for v in range(vmin, vmax + 1):
        S += (-1.0) ** (v + j2 + m2) * Fraction(
            f(j2 + j3 + m1 - v) * f(j1 - m1 + v),
            f(v) * f(j3 - j1 + j2 - v) * f(j3 + m3 - v) * f(v + j1 - j2 - m3),
        )
    C = C * S
    return C


@functools.lru_cache(maxsize=None)
def _su2_clebsch_gordan(
    j1: Union[int, float], j2: Union[int, float], j3: Union[int, float]
) -> np.ndarray:
    """Calculates the Clebsch-Gordon matrix."""
    import math

    assert isinstance(j1, (int, float))
    assert isinstance(j2, (int, float))
    assert isinstance(j3, (int, float))
    mat = np.zeros((int(2 * j1 + 1), int(2 * j2 + 1), int(2 * j3 + 1)))
    if int(2 * j3) in range(int(2 * abs(j1 - j2)), int(2 * (j1 + j2)) + 1, 2):
        for m1 in (x / 2 for x in range(-int(2 * j1), int(2 * j1) + 1, 2)):
            for m2 in (x / 2 for x in range(-int(2 * j2), int(2 * j2) + 1, 2)):
                if abs(m1 + m2) <= j3:
                    mat[int(j1 + m1), int(j2 + m2), int(j3 + m1 + m2)] = _su2_cg(
                        (j1, m1), (j2, m2), (j3, m1 + m2)
                    )
    return mat / math.sqrt(2 * j3 + 1)


@functools.lru_cache(maxsize=None)
def _so3_clebsch_gordan(l1: int, l2: int, l3: int) -> np.ndarray:
    r"""The Clebsch-Gordan coefficients of the real irreducible representations of :math:`SO(3)`.

    Args:
        l1 (int): the representation order of the first irrep
        l2 (int): the representation order of the second irrep
        l3 (int): the representation order of the third irrep

    Returns:
        np.ndarray: the Clebsch-Gordan coefficients
    """
    C = _su2_clebsch_gordan(l1, l2, l3)
    Q1 = change_basis_real_to_complex(l1)
    Q2 = change_basis_real_to_complex(l2)
    Q3 = change_basis_real_to_complex(l3)
    C = np.einsum("ij,kl,mn,ikn->jlm", Q1, Q2, np.conj(Q3.T), C)

    assert np.all(np.abs(np.imag(C)) < 1e-5)
    return np.real(C)


def change_basis_real_to_complex(l: int) -> np.ndarray:
    # https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    q = np.zeros((2 * l + 1, 2 * l + 1), dtype=np.complex128)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1 / np.sqrt(2)
        q[l + m, l - abs(m)] = -1j / np.sqrt(2)
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1) ** m / np.sqrt(2)
        q[l + m, l - abs(m)] = 1j * (-1) ** m / np.sqrt(2)
    return (
        -1j
    ) ** l * q  # Added factor of 1j**l to make the Clebsch-Gordan coefficients real

def integer_powers(x, max_degree: int):
    """Calculates all integer powers up to max_degree of x along axis -2."""
    # TODO: cumprod = exp(cumsum(log)) has low precision
    return Tensor.exp(
            Tensor.cumsum(
             Tensor.log(
                Tensor.cat(
                        Tensor.ones_like(x),
                        Tensor.repeat(x, [max_degree if i == (len(x.shape) - 2)
                                                    else 1 for i in range(len(x.shape))]),
                        dim=-2,
                        ),
                    ), axis=-2
                )
            )