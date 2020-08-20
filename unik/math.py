"""Math / Linear algebra."""
import tensorflow as tf
import numpy as np
import scipy as sp

import tensorflow.linalg as tfl
import tensorflow.math as tfm
import numpy.linalg as npl
import scipy.linalg as spl

from .magik import tensor_compat, Arg, Val
from . import symbolik
from .types import has_tensor, is_tensor, has_tf_tensor, as_tensor, cast, \
                   result_dtype, dtype
from .shapes import rank, size, length, shape, expand_dims, \
                    reshape, flatten, transpose, set_shape, \
                    concat, stack, tile
from .alloc import ones, range
from .indexing import boolean_mask, gather
from .controlflow import while_loop, cond, map_fn
from .various import name_tensor

py_all = all
py_any = any
py_sum = sum

__all__ = ['cumprod', 'minimum', 'maximum', 'cumsum', 'sum', 'prod',
           'any', 'all', 'min', 'max', 'tensordot', 'round', 'floor',
           'ceil', 'clip', 'sum_iter', 'prod_iter', 'matmul_iter',
           'mean', 'argmin', 'argmax', 'invert_permutation', 'matmul',
           'mm', 'lstsq', 'lmdiv', 'rmdiv', 'svd', 'factorial', 'unique',
           'logm', 'expm', 'permutations', 'cartesian', 'sqrt',
           'sin', 'asin', 'arcsin', 'sinh', 'asinh', 'arcsinh',
           'cos', 'acos', 'arccos', 'cosh', 'acosh', 'arccosh']


# These functions are defined in another file to avoid cross dependencies
from ._math_for_indexing import cumprod, minimum, maximum, sqrt
from ._math_reduce import sum, prod, all, any, mean, cumsum, min, max


@tensor_compat(return_dtype=symbolik.result_dtype(Arg('a'), Arg('b')))
def tensordot(a, b, axes, name=None):
    """Tensor dot product.

    Parameters
    ----------
    a : tensor_like
    b : tensor_like
    axes : () or (2, K) tensor_like
        If scalar_like `K`: sum over last `K` axes of `a` and first `K' of `b`.
        If matrix_like: The first row contains indices for axes of `a` and the
            second row contains indices for the axes of `b`
    name : str, optional
        Name for the operation.

    Returns
    -------
    c : tensor_or_array
        rank(c) = rank(a) + rank(b) - K

    """
    if has_tensor([a, b], 'tf'):
        return_dtype = result_dtype(a, b)
        a = cast(a, return_dtype)
        b = cast(b, return_dtype)
        return tf.tensordot(a, b, axes, name=name)
    else:
        return np.tensordot(a, b, axes)


@tensor_compat(return_dtype=symbolik.dtype(Arg('input')))
def round(input, decimals=0, name=None):
    """Round a tensor to the given number of decimals.

    Rounds half to even. Also known as bankers rounding.

    Parameters
    ----------
    input : tensor_like
        Input tensor.
    decimals : () tensor_like[int], default=0
        Number of decimals to keep.
    name : str, optional
        A name for the operation.

    Returns
    -------
    rounded : tensor_or_array
        Rounded tensor.

    """
    if has_tensor([input, decimals], 'tf'):
        decimals = cast(10 ** decimals, dtype(input))
        input = tf.round(input * decimals) / decimals
        input = name_tensor(input, name)
    else:
        input = np.round(input, decimals)
    return input


@tensor_compat(map_batch=False)
def floor(input, name=None):
    """Floor of a tensor.

    Parameters
    ----------
    input : tensor_like
        Input tensor.
    name : str, optional
        A name for the operation.

    Returns
    -------
    floored : tensor_or_array

    """
    if is_tensor(input, 'tf'):
        input = tf.math.floor(input, name=name)
    else:
        input = np.floor(input)
    return input


@tensor_compat(map_batch=False)
def ceil(input, name=None):
    """Ceiling of a tensor.

    Parameters
    ----------
    input : tensor_like
        Input tensor.
    name : str, optional
        A name for the operation.

    Returns
    -------
    ceiled : tensor_or_array

    """
    if is_tensor(input, 'tf'):
        input = tf.math.ceil(input, name=name)
    else:
        input = np.ceil(input)
    return input


@tensor_compat(return_dtype=symbolik.dtype(Arg('input')))
def clip(input, clip_min, clip_max, name=None):
    """Clip a tensor between two values

    Parameters
    ----------
    input : tensor_like
        Input tensor.
    clip_min : () tensor_like or None
        Minimum value to clip by.
    clip_max : () tensor_like or None
        Maximum value to clip by.
    name : str, optional
        A name for the operation.

    Returns
    -------
    clipped : tensor_or_array

    """
    if has_tensor([input, clip_min, clip_max], 'tf'):
        clip_min = cast(clip_min, dtype(input), keep_none=True)
        clip_max = cast(clip_max, dtype(input), keep_none=True)
        return tf.clip_by_value(input, clip_min, clip_max, name=name)
    else:
        return np.clip(input, clip_min, clip_max)


def sum_iter(iterable, start=0, inplace=True):
    """Compute the product of a series of elements.

    This function works with any type that implements __add__
    (or __iadd__ if inplace is True). In particular, it works with
    tf.Tensor objects.

    Parameters
    ----------
    iterable : series of elements
    start : starting value, default=0
    inplace : bool, default=False

    Returns
    -------
    prod_of_values : product of the elements

    """
    sum_of_values = start
    for value in iterable:
        if inplace:
            sum_of_values += value
        else:
            sum_of_values = sum_of_values + value
    return sum_of_values


def prod_iter(iterable, start=1, inplace=False):
    """Compute the product of a series of elements.

    This function works with any type that implements __mul__
    (or __imul__ if inplace is True). In particular, it works with
    tf.Tensor objects.

    Parameters
    ----------
    iterable : series of elements
    start : starting value, default=1
    inplace : bool, default=False

    Returns
    -------
    prod_of_values : product of the elements

    """
    prod_of_values = start
    for value in iterable:
        if inplace:
            prod_of_values *= value
        else:
            prod_of_values = prod_of_values * value
    return prod_of_values


def matmul_iter(iterable, start=None):
    """Compute the matrix product of a series of elements.

    This function works with any type that implements __mul__
    (or __imul__ if inplace is True). In particular, it works with
    tf.Tensor objects.

    Parameters
    ----------
    iterable : series of elements
    start : starting value, default=eye

    Returns
    -------
    prod_of_values : matrix product of the elements

    """
    accumulation = start
    for value in iterable:
        accumulation = value if accumulation is None else accumulation @ value
    return accumulation


@tensor_compat(return_dtype=Arg('dtype'))
def argmin(input, axis=None, dtype='int64', name=None):
    """Index of the minimum value along an axis.

    Parameters
    ----------
    input : tensor_like
        Input tensor.
    axis : () tensor_like, default=None
        Axis along which to extract the minimum index.
        If None, work on the flattened tensor.
    dtype : str or type, default='int64'
        Data type of the returned index.
    name : str, optional
        Name for the operation.

    Returns
    -------
    arg : tensor[dtype] or array[dtype]
        Index of the minimum value along an axis.


    """
    if is_tensor(input, 'tf'):
        return tf.math.argmin(input, axis=axis, output_type=dtype, name=name)
    else:
        input = np.argmin(input, axis=axis)
        if dtype is not None:
            input = cast(input, dtype)
        return input


@tensor_compat(return_dtype=Arg('dtype'))
def argmax(input, axis=None, dtype='int64', name=None):
    """Index of the maximum value along an axis.

    Parameters
    ----------
    input : tensor_like
        Input tensor.
    axis : () tensor_like, default=None
        Axis along which to extract the maximum index.
        If None, work on the flattened tensor.
    dtype : str or type, default='int64'
        Data type of the returned index.
    name : str, optional
        Name for the operation.

    Returns
    -------
    arg : tensor[dtype] or array[dtype]
        Index of the maximum value along an axis.


    """
    if is_tensor(input, 'tf'):
        return tf.math.argmax(input, axis=axis, output_type=dtype, name=name)
    else:
        input = np.argmax(input, axis=axis)
        if dtype is not None:
            input = cast(input, dtype)
        return input


@tensor_compat
def invert_permutation(perm, name=None):
    """Invert a permutation.

    Parameters
    ----------
    perm : (length, ) tensor_like
        A permutation vector. E.g., `[0, 2, 1]`.
    name : str, optional
        A name for the operation.

    Returns
    -------
    iperm : (length, ) tensor_or_array
        The inverse permutation.

    """
    if has_tf_tensor(perm):
        return tf.math.invert_permutation(perm, name=name)
    else:
        # https://stackoverflow.com/questions/11649577/
        perm = as_tensor(perm)
        iperm = np.empty_like(perm)
        iperm[perm] = np.arange(perm.size)
        return iperm


@tensor_compat(return_dtype=symbolik.result_dtype(Arg('a'), Arg('b')))
def matmul(a, b, transpose_a=False, transpose_b=False,
           adjoint_a=False, adjoint_b=False, name=None):
    """Matrix multiplication between (fields of) matrices.

    Parameters
    ----------
    a : (..., N, K) tensor_like
        Left matrix.
    b : (..., K, M) tensor_like
        Right matrix.
    transpose_a : bool, default=False
        If True, transpose the last two dimensions of ``a``.
    transpose_b : bool, default=False
        If True, transpose the last two dimensions of ``b``.
    adjoint_a : bool, default=False
        If True, conjugate transpose the last two dimensions of ``a``.
    adjoint_b : bool, default=False
        If True, conjugate transpose the last two dimensions of ``b``.
    name : str, optional
        A name for the operation.

    Returns
    -------
    prod : (..., N, M) tensor_or_array
        Matrix product ``a @ b``.

    Broadcasting rules
    ------------------
    * Dimensions of length 1 may be prepended to either tensor.
    * Tensors may be repeated along dimensions of length 1.

    """
    if has_tensor([a, b], 'tf'):
        return_dtype = result_dtype(a, b)
        a = cast(a, return_dtype)
        b = cast(b, return_dtype)
        a = expand_dims(a, axis=0, ndim=maximum(0, rank(b)-rank(a)))
        b = expand_dims(b, axis=0, ndim=maximum(0, rank(a)-rank(b)))
        return tf.linalg.matmul(a, b, transpose_a, transpose_b,
                                adjoint_a, adjoint_b, name=name)
    else:
        a = expand_dims(a, axis=0, ndim=maximum(0, rank(b)-rank(a)))
        b = expand_dims(b, axis=0, ndim=maximum(0, rank(a)-rank(b)))
        if adjoint_a:
            perm = concat((range(rank(a)-2), [-1, -2]))
            a = np.conjugate(np.transpose(a, perm))
        elif transpose_a:
            perm = concat((range(rank(a)-2), [-1, -2]))
            a = np.transpose(a, perm)
        if adjoint_b:
            perm = concat((range(rank(b)-2), [-1, -2]))
            b = np.conjugate(np.transpose(b, perm))
        elif transpose_b:
            perm = concat((range(rank(b)-2), [-1, -2]))
            b = np.transpose(b, perm)
        return np.matmul(a, b)


"""Alias for matmul."""
mm = matmul


@tensor_compat(return_dtype=symbolik.result_dtype(Arg('a'), Arg('b')))
def matvec(a, b, transpose_a=False, adjoint_a=False, name=None):
    """Matrix multiplication between (fields of) matrices.

    Parameters
    ----------
    a : (..., N, K) tensor_like
        Left matrix.
    b : (..., K) tensor_like
        Right matrix.
    transpose_a : bool, default=False
        If True, transpose the last two dimensions of ``a``.
    adjoint_a : bool, default=False
        If True, conjugate transpose the last two dimensions of ``a``.
    name : str, optional
        A name for the operation.

    Returns
    -------
    prod : (..., N) tensor_or_array
        Matrix-vector product ``a @ b``.

    Broadcasting rules
    ------------------
    * Dimensions of length 1 may be prepended to either tensor.
    * Tensors may be repeated along dimensions of length 1.

    """
    if has_tensor([a, b], 'tf'):
        return_dtype = result_dtype(a, b)
        a = cast(a, return_dtype)
        b = cast(b, return_dtype)
        a = expand_dims(a, axis=0, ndim=maximum(0, rank(b)+1-rank(a)))
        b = expand_dims(b, axis=0, ndim=maximum(0, rank(a)-rank(b)-1))
        return tf.linalg.matvec(a, b, transpose_a, adjoint_a, name=name)
    else:
        a = expand_dims(a, axis=0, ndim=maximum(0, rank(b)+1-rank(a)))
        b = expand_dims(b, axis=0, ndim=maximum(0, rank(a)-rank(b)-1))
        if adjoint_a:
            perm = concat((range(rank(a)-2), [-1, -2]))
            a = np.conjugate(np.transpose(a, perm))
        elif transpose_a:
            perm = concat((range(rank(a)-2), [-1, -2]))
            a = np.transpose(a, perm)
        return np.matmul(a, b[..., None])[..., 0]


@tensor_compat(return_dtype=symbolik.result_dtype(Arg('a'), Arg('b')))
def lstsq(a, b, l2_regularizers=None, rcond=None, name=None):
    r"""Least-square solution of a (field of) linear systems.

    Parameters
    ----------
    a : (..., M, N) tensor_like
        Left matrix.

        .. np:: Fields are not vectorized. Expect something slow.

    b : (..., M, K) tensor_like
        Right matrix.

        .. np:: Fields are not vectorized. Expect something slow.

    l2_regularizers : () tensor_like[float], default=None
        .. tf:: If not `None` an algorithm based on the
                numerically robust complete orthogonal decomposition is
                used. This computes the minimum-norm least-squares
                solution, even when \\(A\\) is rank deficient. This path
                is typically 6-7 times slower than the fast path
                (when `l2_regularizers` is `None`).
        .. np:: Not used.

    rcond : float, default=None
        .. np:: Cut-off ratio for small singular values of `a`.
                For the purposes of rank determination, singular values
                are treated as zero if they are smaller than `rcond`
                times the largest singular value of `a`.
                * If -1: use the machine precision.
                * If None: use the machine precision times `max(M, N)`
        .. tf:: Not used.

    name : str, optional
        Name for the operation.

    Returns
    -------
    x : (..., N, K) tensor_or_array
        Solution of the linear system, e.g., `a \ b`

    """
    if has_tf_tensor([a, b, l2_regularizers]):
        if l2_regularizers is None:
            l2_regularizers = 0.0
            fast = False
        else:
            fast = True
        l2_regularizers = cast(l2_regularizers, 'double')
        return_dtype = result_dtype(a, b)
        a = cast(a, return_dtype)
        b = cast(b, return_dtype)
        return tf.linalg.lstsq(a, b, l2_regularizers=l2_regularizers,
                               fast=fast, name=name)
    else:
        a = expand_dims(a, maximum(rank(b)-rank(a), 0))
        b = expand_dims(a, maximum(rank(a)-rank(b), 0))
        a_shape = shape(a)
        a_mat_shape = a_shape[-2:]
        a_shape = a_shape[:-2]
        b_shape = shape(b)
        b_mat_shape = b_shape[-2:]
        b_shape = b_shape[:-2]
        shape_compat = [(sa == sb or sa == 1 or sb == 1)
                        for sa, sb in zip(a_shape, b_shape)]
        if not py_all(shape_compat):
            raise ValueError('shape mismatch: objects cannot be broadcast '
                             'to a single shape')
        out_shape = tuple(maximum(sa, sb) for sa, sb in zip(a_shape, b_shape))
        out_mat_shape = (shape(a)[-1], shape(b)[-1])
        tile_a = [so//sa for so, sa in zip(out_shape, a_shape)] + (1, 1)
        tile_b = [so//sb for so, sb in zip(out_shape, b_shape)] + (1, 1)
        a = reshape(tile(a, tile_a), (-1,) + a_mat_shape)
        b = reshape(tile(b, tile_b), (-1,) + b_mat_shape)
        out = map_fn(lambda x: np.linalg.lstsq(x[0], x[1], rcond=rcond)[0],
                     [a, b])
        out = reshape(out, out_shape + out_mat_shape)
        return out


@tensor_compat(return_dtype=symbolik.result_dtype(Arg('a'), Arg('b')))
def lmdiv(a, b, l2_regularizers=None, rcond=None, name='lmdiv'):
    r"""Left matrix division A\B.

    Parameters
    ----------
    a : (M, [N]) tensor_like
    b : (M, [K]) tensor_like

    l2_regularizers : () tensor_like[float], default=None
        See `lstsq`.
    rcond : float, default=None
        See `lstsq`.
    name : str, optional
        Name for the operation.

    Returns
    -------
    X : (N, [K]) tensor_or_array

    """
    a = as_tensor(a)
    b = as_tensor(b)
    a = cond(rank(a) == 1, lambda: a[..., None], lambda: a)
    x = lstsq(a, b, l2_regularizers=l2_regularizers, rcond=rcond, name=name)
    return x


@tensor_compat(return_dtype=symbolik.result_dtype(Arg('a'), Arg('b')))
def rmdiv(a, b, l2_regularizers=None, rcond=None, name='rmdiv'):
    r"""Right matrix division A/B.

    Parameters
    ----------
    a : (M, [N]) tensor_like
    b : (K, [N]) tensor_like

    l2_regularizers : () tensor_like[float], default=None
        See `lstsq`.
    rcond : float, default=None
        See `lstsq`.
    name : str, optional
        Name for the operation.

    Returns
    -------
    x : (M, K) tensor_or_array

    """
    a = as_tensor(a)
    b = as_tensor(b)
    a = cond(rank(a) == 1, lambda: a[..., None], lambda: a)
    b = cond(rank(b) == 1, lambda: b[..., None], lambda: b)
    a = transpose(a)
    b = transpose(b)
    return transpose(lstsq(b, a, l2_regularizers=l2_regularizers, rcond=rcond),
                     name=name)


@tensor_compat
def svd(input, full_matrices=True, compute_uv=True,
        hermitian=False, name=None):
    """Singular value decomposition.

    Parameters
    ----------
    input - (..., M, N) tensor_like
        Input (field of) matrices. Let P = minimum(M, N).

    full_matrices - bool, default=True
        If True, return full U and V matrices. Else, truncate columns at P.

        .. warning:: default is True (same as np // different from tf)

    compute_uv : bool, default=True
        If True, return U, S, V. Else return S.

    hermitian : bool, default=False
        .. np:: Assume that the input matrices are hermitian.
        .. tf:: Not used.

    name : str, optional
        Name for the operation.

    Returns
    -------
    U : (..., M, M) or  (..., M, P) tensor, if `compute_uv is True`
        Left singular vectors.
    S : (..., P) tensor
        Singular values.
    V : (..., N, N) or  (..., N, P) tensor, if `compute_uv is True`
        Right singular vectors.

    ..warning:: Order is U, S, V (same as numpy), while tf's order is S, U, V.

    """
    input = as_tensor(input)
    if is_tensor(input, 'tf'):
        usv = tf.linalg.svd(input,
                            full_matrices=full_matrices,
                            compute_uv=compute_uv,
                            name=name)
        return usv if not compute_uv else (usv[1], usv[0], usv[2])
    else:
        return np.linalg.svd(input,
                             full_matrices=full_matrices,
                             compute_uv=compute_uv,
                             hermitian=hermitian,
                             name=name)


@tensor_compat
def factorial(x, name=None):
    """Factorial of a number.

    .. tf:: Uses the exponential of log-gamma to compute the factorial.
            This may lead to inaccuracies.

    Parameters
    ----------
    x : tensor_like
    name : str, optional

    Returns
    -------
    factorial : tensor_or_array

    """
    if is_tensor(x, 'tf'):
        input_dtype = x.dtype
        float_types = (tf.dtypes.float16, tf.dtypes.float32, tf.dtypes.float64)
        if input_dtype not in float_types:
            x = cast(x, 'float32')
        x = tf.where(x == 0,
                     tf.ones_like(x, shape=[1] * rank(x)),
                     tf.math.exp(tf.math.lgamma(x)))
        x = cast(x, input_dtype)
        return name_tensor(x, name)
    else:
        return np.math.factorial(x)


@tensor_compat
def unique(input, return_index=False, return_inverse=False,
           return_counts=False, index_dtype='int32', name=None):
    """Return the unique values from a vector.

    Parameters
    ----------
    input : (L,) tensor_like
        Input vector.
    return_index : bool, default=False
        Also return the index of the first apparition of each element.
        .. tf:: Not implemented.
    return_inverse : bool, default=False
        Also return indices that allow the initial vector to be rebuilt.
    return_counts : bool, default=False
        Also return the number of apparitions of each element.
    index_dtype : str or type, default='int32'
        Data type of the returned indices.
    name : str, optional
        A name for the operation.

    Returns
    -------
    unique : (K,) tensor_or_array
        Unique elements of the input vector, ordered by first apparition.
    index : (K,) tensor[index_dtype] or array[index_dtype]
        First index of each unique element in the input vector.
    inverse : (L,) tensor[index_dtype] or array[index_dtype]
        Inverse mapping, such that `input = unique[inverse'
    counts : (K,) tensor[index_dtype] or array[index_dtype]
        Number of apparitions of each unique element.

    """
    if has_tensor(input, 'tf'):
        if return_counts:
            values, indices, counts = \
                tf.unique_with_counts(input, out_idx=index_dtype, name=name)
        else:
            values, indices = tf.unique(input, out_idx=index_dtype, name=name)
        output = [values]
        if return_index:
            raise NotImplementedError
        if return_inverse:
            output.append(indices)
        if return_counts:
            output.append(counts)
    else:
        output = np.unique(input,
                           return_index=return_index,
                           return_inverse=return_inverse,
                           return_counts=return_counts)
        output = list(output)
        output[1:] = [cast(e, index_dtype) for e in output[1:]]
    return output


@tensor_compat
def cartesian(input, shape=None, flatten=True):
    """ Cartesian product: P = A * B * ...

    Parameters
    ----------
    input : vector_like or (*shape) tensor_like[vector_like]
        Values to sample to build the cartesian product.

    shape : int or vector_like[int], optional
        If not None, it represents the output shape, while input
        represents the set of possible values, which is shared across
        elements.

    flatten : bool, default=True
        If true, flatten the *arrangements part of the output shape.

    Returns
    -------
    cart : (*arrangements, *shape) tensor_or_array
        All possible arrangements of values in the input ranges,
        laid out as specified by the input shape.

    Examples
    --------
    >>> from unik import cartesian
    >>> import numpy as np
    >>> # Cartesian product
    >>> A = [1, 2]
    >>> B = [3, 4, 5]
    >>> AxB = cartesian([A, B])
    >>> check = [[1, 3],
    >>>          [1, 4],
    >>>          [1, 5],
    >>>          [2, 3],
    >>>          [2, 4],
    >>>          [2, 5]]
    >>> assert(np.all(np.asarray(AxB) == np.asarray(check)))
    >>> # Cartesian power
    >>> A = [1, 2]
    >>> AxA = cartesian(A, shape=[2])
    >>> check = [[1, 1],
    >>>          [1, 2],
    >>>          [2, 1],
    >>>          [2, 2]]
    >>> assert(np.all(np.asarray(AxA) == np.asarray(check)))

    """
    # If an output shape is provided, we are in the case of a cartesian power
    if shape is not None:
        return _cart_power(input, shape, flatten)
    else:
        return _cart_product(input, flatten)


def _cart_power(input, _shape, _flatten):
    """Cartesian power: P = A ** n = A * A * ..."""
    # Based on:
    # https://gist.github.com/andrewcsmith/edbd3bd8ea4c575685a4348299944383

    # shape(a) == [*input_shape]
    input = as_tensor(input)
    input_shape = shape(input)
    input = flatten(input)
    numel = prod(_shape)

    # shape(a) == [numel]

    def _body(a, b, numel):
        # shape(a) == [A, K]
        # shape(b) == [B, 1]
        nb_tile_a = stack([1, shape(b)[0], 1])
        nb_tile_b = stack([shape(a)[0], 1, 1])
        tile_a = tile(a[:, None, :], nb_tile_a)  # shape(tile_a) == [A, B, K]
        tile_b = tile(b[None, :, :], nb_tile_b)  # shape(tile_b) == [A, B, 1]

        ab = concat([tile_a, tile_b], axis=-1)  # shape(ab) == [A, B, K+1]
        ab = reshape(ab, [-1, shape(a)[-1] + shape(b)[-1]])
        # shape(ab) = [A*B, K+1]
        return ab, b, numel

    def _cond(a, b, numel):
        return cast(shape(a)[-1], 'int64') < cast(numel, 'int64')

    input = input[..., None]
    shape_invariants = [tf.TensorShape([None, None]),
                        tf.TensorShape([None, 1]),
                        tf.TensorShape([])]
    output = while_loop(_cond, _body, [input, input, numel],
                        shape_invariants=shape_invariants,
                        parallel_iterations=1)[0]
    output_shape = [-1] if _flatten else tile(input_shape, [numel])
    _shape = cond(rank(_shape) == 0,
                  lambda: expand_dims(_shape, 0),
                  lambda: _shape)
    output = reshape(output, concat((output_shape, _shape)))
    return output


def _cart_product(input, _flatten):
    """Cartesian product: P = A * B * ..."""

    def _is_leaf(input):
        """True if element does not have children."""
        if isinstance(input, list) or isinstance(input, tuple):
            return False
        if tf.is_tensor(input) or isinstance(input, np.ndarray):
            return rank(input) == 0
        else:
            return True

    def _is_last(input):
        """True if all children are leaves."""
        if isinstance(input, list) or isinstance(input, tuple):
            return any(_is_leaf(i) for i in input)
        if tf.is_tensor(input) or isinstance(input, np.ndarray):
            return rank(input) == 1
        else:
            return None

    def _get_shape(input):
        """Compute the 'shape' of a nested list."""
        list_shape = []
        if isinstance(input, list) or isinstance(input, tuple):
            list_shape.append(len(input))
            is_last = [_is_last(i) for i in input]
            if not any(is_last):
                subshapes = [_get_shape(i) for i in input]
                subdim = set(s[0] for s in subshapes)
                if len(subdim) > 1:
                    raise ValueError('Input shape not consistent. '
                                     'Only the last dimension can have '
                                     'non-constant size.')
                list_shape += subshapes[0]
            elif not all(is_last):
                raise ValueError('Leaf and non-leaf elements at the '
                                 'same level.')
        elif tf.is_tensor(input) or isinstance(input, np.ndarray):
            list_shape += list(shape(input)[-1])
        return list_shape

    def _flatten_structure(input):
        """Flatten a nested list."""
        output = []
        if isinstance(input, list) or isinstance(input, tuple):
            is_last = [_is_last(i) for i in input]
            if any(is_last):
                output = output + input
            else:
                for i in input:
                    output = output + _flatten_structure(i)
        elif tf.is_tensor(input) or isinstance(input, np.ndarray):
            input = reshape(input, [-1, shape(input)[-1]])
            output = concat((input, output))
        return output

    def _body(a, b):
        # a.shape == [*shape_a, K]
        # b.shape == [*shape_b, 1]
        nb_tile_a = concat((ones(rank(a) - 1, 'int'), shape(b)[:-1], [1]))
        nb_tile_b = concat((shape(a)[:-1], ones(rank(b), 'int')))
        tile_a = expand_dims(a, -2, ndim=rank(b) - 1)
        tile_a = tile(tile_a, nb_tile_a)  # [*a.shape, *b.shape, K]
        tile_b = expand_dims(b, 0, ndim=rank(a) - 1)
        tile_b = tile(tile_b, nb_tile_b)  # [*a.shape, *b.shape, 1]

        ab = concat([tile_a, tile_b], axis=-1)  # [*a.shape, *b.shape, K+1]
        return ab

    # Compute info on output structure
    input_shape = _get_shape(input)
    input = _flatten_structure(input)
    nb_input = length(input)

    a, *input = input
    a = as_tensor(a)[..., None]
    while length(input) > 0:
        b, *input = input
        b = as_tensor(b)[..., None]
        a = _body(a, b)
    if _flatten:
        a = reshape(a, [-1, nb_input])
    a = reshape(a, concat((shape(a)[:-1], input_shape)))
    return a


@tensor_compat
def permutations(input, shape=None):
    """Return all possible permutations of a set.

    Parameters
    ----------
    input : (input_length,) vector_like
        Input set in which values are sampled.
    shape : int or vector_like[int], default=input_length
        If not None, it represents the output shape, while input
        represents the set of possible values, which is shared across
        elements.


    Returns
    -------
    perm : (output_length, *shape) tensor_or_array
        All possible permutations of the input set.
        `output_length` = TODO

    """

    return _permutations(input, shape)


def _permutations(input, output_shape):
    # https://stackoverflow.com/questions/104420/

    def unique_count_per_row(x):
        """Compute the number of unique elements in each row of a matrix

        Parameters
        ----------
        x : (K, N) matrix_like

        Returns
        -------
        count : (K,) vector_like[int]

        """

        def _body(a, b):
            # shape(a) == [k]       (already processed rows)
            # shape(b) == [K-k, N]  (remaining rows)
            b0 = b[0, ...]
            b = b[1:, ...]
            b0 = size(unique(b0, index_dtype='int32'))
            a = concat((a, b0[None]))
            return a, b

        def _cond(a, b):
            return length(b) > 0

        # shape(x) == [K, N]
        input_length = shape(x)[1]
        if is_tensor(input_length, 'tf'):
            input_length = None
        shape_invariants = [tf.TensorShape([None]),
                            tf.TensorShape([None, input_length])]
        return while_loop(_cond, _body, [as_tensor([], dtype='int32'), x],
                          shape_invariants=shape_invariants,
                          parallel_iterations=1)[0]

    # Compute shapes
    input = flatten(input)
    input_length = length(input)
    output_shape = [input_length] if output_shape is None else flatten(
        output_shape)
    output_length = prod(output_shape)
    nb_perm = factorial(input_length) * factorial(input_length - output_length)

    # Cartesian product of indices
    indices = cartesian(range(input_length), shape=output_shape)

    # Select only permutations (= discard rows with repeats)
    mask_indices = unique_count_per_row(indices) == shape(indices)[1]
    indices = boolean_mask(indices, mask_indices, axis=0)
    indices = set_shape(indices, [nb_perm, shape(indices)[1]])

    # Use indices to extract permutations of elements from input
    perms = gather(input, indices)

    # Final reshape
    output_shape = concat(([shape(indices)[0]], output_shape))
    perms = reshape(perms, output_shape)
    return perms


@tensor_compat(map_batch=False)
def exp(input, name=None):
    """Element-wise exponential of a tensor."""
    if is_tensor(input, 'tf'):
        return tf.math.exp(input, name=name)
    else:
        return np.exp(input)


@tensor_compat(map_batch=False)
def log(input, name=None):
    """Element-wise natural logarithm of a tensor."""
    if is_tensor(input, 'tf'):
        return tf.math.log(input, name=name)
    else:
        return np.log(input)


@tensor_compat(map_batch=False)
def sin(input, name=None):
    """Element-wise sine of a tensor."""
    if is_tensor(input, 'tf'):
        return tf.math.sin(input, name=name)
    else:
        return np.sin(input)


@tensor_compat(map_batch=False)
def sinh(input, name=None):
    """Element-wise hyperbolic sine of a tensor."""
    if is_tensor(input, 'tf'):
        return tf.math.sinh(input, name=name)
    else:
        return np.sinh(input)


@tensor_compat(map_batch=False)
def asin(input, name=None):
    """Element-wise arc-sine of a tensor."""
    if is_tensor(input, 'tf'):
        return tf.math.asin(input, name=name)
    else:
        return np.arcsin(input)


arcsin = asin


@tensor_compat(map_batch=False)
def asinh(input, name=None):
    """Element-wise hyperbolic arc-sine of a tensor."""
    if is_tensor(input, 'tf'):
        return tf.math.asinh(input, name=name)
    else:
        return np.arcsinh(input)


arcsinh = asinh


@tensor_compat(map_batch=False)
def cos(input, name=None):
    """Element-wise cosine of a tensor."""
    if is_tensor(input, 'tf'):
        return tf.math.cos(input, name=name)
    else:
        return np.cos(input)


@tensor_compat(map_batch=False)
def cosh(input, name=None):
    """Element-wise hyperbolic cosine of a tensor."""
    if is_tensor(input, 'tf'):
        return tf.math.cosh(input, name=name)
    else:
        return np.cosh(input)


@tensor_compat(map_batch=False)
def acos(input, name=None):
    """Element-wise arc-cosine of a tensor."""
    if is_tensor(input, 'tf'):
        return tf.math.acos(input, name=name)
    else:
        return np.arccos(input)


arccos = acos


@tensor_compat(map_batch=False)
def acosh(input, name=None):
    """Element-wise hyperbolic arc-cosine of a tensor."""
    if is_tensor(input, 'tf'):
        return tf.math.acosh(input, name=name)
    else:
        return np.arccosh(input)


arccosh = acosh


@tensor_compat(map_batch=False)
def expm(input, name=None):
    """Matrix exponential of (a field of) tensors / arrays.

    Parameters
    ----------
    input : (..., M, M) tensor_like[float or complex]
        Input field of matrices.
    name : str, optional
        A name for the operation.

    Returns
    -------
    output : (..., M, M) tensor_or_array
        Output field of matrices, with same data type as the input.

    Notes
    -----
    ..  Applying this function to fields of numpy matrices is quite
        slow, as it is not vectorized.

    """
    if is_tensor(input, 'tf'):
        return tf.linalg.expm(input, name=name)
    else:
        in_shape = list(shape(input))
        mat_shape = in_shape[-2:]
        input = reshape(input, [-1] + mat_shape)
        input = map_fn(spl.expm, input)
        input = reshape(input, in_shape)
        return input


@tensor_compat(map_batch=False)
def logm(input, name=None):
    """Matrix logarithm of (a field of) tensors / arrays.

    Parameters
    ----------
    input : (..., M, M) tensor_like[float or complex]
        Input field of matrices.
    name : str, optional
        A name for the operation.

    Returns
    -------
    output : (..., M, M) tensor_or_array
        Output field of matrices, with same data type as the input.

    Notes
    -----
    ..  Applying this function to fields of numpy matrices is quite
        slow, as it is not vectorized.
    ..  In tensorflow, it is only implemented for complex matrices.
        When applied to real matrices, two implicit conversions are
        therefore performed, before and after the logarithm.

    """
    if is_tensor(input, 'tf'):
        in_dtype = input.dtype
        if in_dtype == 'float32':
            input = cast(input, 'complex64')
        elif in_dtype == 'float64':
            input = cast(input, 'complex128')
        input = tf.linalg.logm(input)
        input = cast(input, in_dtype)
        return name_tensor(input, name)
    else:
        in_shape = list(shape(input))
        mat_shape = in_shape[-2:]
        input = reshape(input, [-1] + mat_shape)
        input = map_fn(spl.logm, input)
        input = reshape(input, in_shape)
        return input
