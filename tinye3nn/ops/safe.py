import functools
from typing import Any, Optional, Tuple, Union
from tinygrad import Tensor


def norm(
    x,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    ):
    """L2-norm of x along the specified axis.

    This functions is equivalent to calling jax.numpy.linalg.norm(x, axis=axis,
    keepdim=keepdim), with the exception that it is more numerically stable
    (very large absolute values in x will not cause overflow and very small
    absolute values in x will still lead to stable and well-defined derivatives).

    Args:
      x: Input array.
      axis: Axis or axes along which the L2-norm is computed. The default,
        axis=None, will compute the norm of all elements of the input array (as if
        it was one large vector). If axis is negative it counts from the last to
        the first axis.
      keepdim: If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result will
        broadcast correctly against the input array.

    Returns:
      The L2-norm of x along axis. An array with the same shape as x, with the
      specified axis removed. If x is a 0-d array, or if axis is None, a scalar is
      returned.
    """
    # TODO: Need to implement
    # https://github.com/atomicarchitects/BAEsislib/blob/2213e6da35b322af60f2593819e71a0eec35a5f2/BAEsislib/ops/safe.py#L33
    a = Tensor.max(Tensor.abs(x))
    b = x / a
    return a * Tensor.sqrt(Tensor.sum(b * b, axis=axis, keepdim=keepdim))


def normalize_and_return_norm(
    x,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
):
    """Normalize x using the L2-norm along the specified axis and return its norm.

    If x has a norm of almost zero, it is left unnormalized, because normalization
    becomes numerically unstable.

    Args:
      x: Input array.
      axis: Axis or axes along which the L2-norm is computed. The default,
        axis=None, will compute the norm of all elements of the input array (as if
        it was one large vector). If axis is negative it counts from the last to
        the first axis.
      keepdim: If this is set to True, the axes which are reduced for computing
        the norm are left in the norm result as dimensions with size one.

    Returns:
      A tuple consisting of the normalized array and its norm.
    """
    n = norm(x, axis=axis, keepdim=True)
    # TODO: If n * n is smaller than finfo.tiny, the gradient becomes unstable.
    y = x / n
    if not keepdim:
        n = Tensor.squeeze(n, axis=axis)
    return y, n


def normalize(
    x, axis: Optional[Union[int, Tuple[int, ...]]] = None
):
    """Normalize x using the L2-norm of x along the specified axis.

    If x has a norm of almost zero, it is left unnormalized, because normalization
    becomes numerically unstable.

    Args:
      x: Input array.
      axis: Axis or axes along which the L2-norm is computed. The default,
        axis=None, will compute the norm of all elements of the input array (as if
        it was one large vector). If axis is negative it counts from the last to
        the first axis.

    Returns:
      The normalized array with the same shape as x.
    """
    y, _ = normalize_and_return_norm(x, axis=axis, keepdim=True)
    return y