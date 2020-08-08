"""Math / Linear algebra."""
import tensorflow as tf
import numpy as np
import scipy as sp

from .magik import tensor_compat
from .types import has_tensor, is_tensor, has_tf_tensor, as_tensor, cast
from .shapes import rank, size, length, shape, expand_dims, \
                    reshape, flatten, transpose, set_shape, \
                    concat, stack, tile
from .alloc import ones
from .indexing import boolean_mask, gather
from .controlflow import while_loop, cond
from .various import name_tensor


# These functions are defined in another file to avoid cross dependencies
from ._math_for_indexing import cumprod, minimum, maximum


@tensor_compat
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
    c : tensor or array
        rank(c) = rank(a) + rank(b) - K

    """
    if tf.executing_eagerly() or tf.is_tensor(a) or tf.is_tensor(b):
        return tf.tensordot(a, b, axes, name=name)
    else:
        return np.tensordot(a, b, axes)


@tensor_compat
def round(input, decimals=0, name=None):
    """Round a tensor / array."""
    if tf.is_tensor(input) or tf.is_tensor(decimals):
        decimals = 10 ** decimals
        input = tf.round(input * decimals) / decimals
        input = name_tensor(input, name)
    else:
        input = np.round(input, decimals)
    return input


@tensor_compat(map_batch=False)
def floor(input, name=None):
    """Floor a tensor / array."""
    if tf.is_tensor(input):
        input = tf.math.floor(input, name=name)
    else:
        input = np.floor(input)
    return input


@tensor_compat(map_batch=False)
def ceil(input, name=None):
    """Ceil a tensor / array."""
    if tf.is_tensor(input):
        input = tf.math.ceil(input, name=name)
    else:
        input = np.ceil(input)
    return input


@tensor_compat
def clip(input, clip_min, clip_max, name=None):
    """Clip a tensor / array."""
    if tf.is_tensor(input) or tf.is_tensor(clip_min) or tf.is_tensor(clip_max):
        input = tf.clip_by_value(input, clip_min, clip_max, name=name)
    else:
        input = np.clip(input, clip_min, clip_max)
    return input


@tensor_compat
def sum(input, axis=None, keepdims=False, name=None):
    """Sum across an axis"""
    if has_tf_tensor(input):
        return tf.math.reduce_sum(input, axis, keepdims=keepdims, name=name)
    else:
        return np.sum(input, axis, keepdims=keepdims)


@tensor_compat
def prod(input, axis=None, keepdims=False, name=None):
    """Product across an axis"""
    if has_tf_tensor(input):
        return tf.math.reduce_prod(input, axis, keepdims=keepdims, name=name)
    else:
        return np.prod(input, axis, keepdims=keepdims)


@tensor_compat
def cumsum(input, axis=None, dtype=None, exclusive=False, reverse=False,
           name=None):
    """Cumulative sum across an axis."""
    if has_tf_tensor(input):
        input = cast(input, dtype)
        if axis is None:
            input = tf.reshape(input, [-1])
            axis = 0
        return tf.math.cumsum(input, axis, exclusive=exclusive,
                              reverse=reverse, name=name)
    else:
        input = np.asarray(input)
        if axis is None:
            input = input.flatten()
            axis = 0
        if reverse:
            input = np.flip(input, axis=axis)
        if exclusive:
            input = np.take(input, range(shape(input)[axis] - 1), axis=axis)
        input = np.cumsum(input, axis, dtype=dtype)
        if exclusive:
            pad = np.zeros((input.dim, 2), dtype='int')
            pad[axis, 0] = 1
            input = np.pad(input, pad, constant_values=1)
        if reverse:
            input = np.flip(input, axis)
        return input


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


@tensor_compat
def min(input, axis=None, keepdims=False, name=None):
    """Minimum of a tensor / array along an axis."""
    if tf.is_tensor(input):
        return tf.math.reduce_min(input, axis, keepdims, name)
    else:
        return np.min(input, axis=axis, keepdims=keepdims)


@tensor_compat
def max(input, axis=None, keepdims=False, name=None):
    """Maximum of a tensor / array along an axis."""
    if tf.is_tensor(input):
        return tf.math.reduce_max(input, axis, keepdims, name)
    else:
        return np.max(input, axis=axis, keepdims=keepdims)


@tensor_compat
def argmin(input, axis=None, dtype='int64', name=None):
    """Index of the minimum value along an axis."""
    if is_tensor(input, 'tf'):
        return tf.math.argmin(input, axis=axis, output_type=dtype, name=name)
    else:
        input = np.argmin(input, axis=axis)
        if dtype is not None:
            input = cast(input, dtype)
        return input


@tensor_compat
def argmax(input, axis=None, dtype='int64', name=None):
    """Index of the maximum value along an axis."""
    if is_tensor(input, 'tf'):
        return tf.math.argmax(input, axis=axis, output_type=dtype, name=name)
    else:
        input = np.argmax(input, axis=axis)
        if dtype is not None:
            input = cast(input, dtype)
        return input


@tensor_compat
def invert_permutation(perm, name=None):
    """Invert a permutation."""
    if has_tf_tensor(perm):
        return tf.math.invert_permutation(perm, name=name)
    else:
        # https://stackoverflow.com/questions/11649577/
        perm = as_tensor(perm)
        iperm = np.empty_like(perm)
        iperm[perm] = np.arange(perm.size)
        return iperm


@tensor_compat
def matmul(a, b, transpose_a=False, transpose_b=False,
           adjoint_a=False, adjoint_b=False, name=None):
    """Matrix multiplication between two tensors / arrays."""
    if has_tf_tensor([a, b]):
        return tf.linalg.matmul(a, b, transpose_a, transpose_b,
                                adjoint_a, adjoint_b, name=name)
    else:
        if adjoint_a:
            a = np.conjugate(np.transpose(a))
        elif transpose_a:
            a = np.transpose(a)
        if adjoint_b:
            b = np.conjugate(np.transpose(b))
        elif transpose_b:
            b = np.transpose(b)
        return np.matmul(a, b)


"""Alias for matmul."""
mm = matmul


@tensor_compat
def lstsq(A, B, l2_regularizers=0.0, fast=True, rcond=None, name=None):
    """Least-square solution of a linear system."""
    if has_tf_tensor([A, B, l2_regularizers]):
        return tf.linalg.lstsq(A, B, l2_regularizers=l2_regularizers,
                               fast=fast, name=name)
    else:
        return np.linalg.lstsq(A, B, rcond=rcond)[0]


@tensor_compat
def lmdiv(A, B, l2_regularizers=0.0, fast=True, rcond=None, name='lmdiv'):
    r"""Left matrix division A\B.

    Parameters
    ----------
    A : (M, [N]) tensor_like
    B : (M, [K]) tensor_like

    Returns
    -------
    X : (N, [K]) tensor or array

    """
    A = as_tensor(A)
    B = as_tensor(B)
    A = cond(rank(A) == 1, lambda: A[..., None], lambda: A)
    X = lstsq(A, B, l2_regularizers=l2_regularizers, fast=fast,
              rcond=rcond, name=name)
    return X


@tensor_compat
def rmdiv(A, B, l2_regularizers=0.0, fast=True, rcond=None, name='rmdiv'):
    r"""Right matrix division A/B.

    Parameters
    ----------
    A : (M, [N]) tensor_like
    B : (K, [N]) tensor_like

    Returns
    -------
    X : (M, K) tensor or array

    """
    A = as_tensor(A)
    B = as_tensor(B)
    A = cond(rank(A) == 1, lambda: A[..., None], lambda: A)
    B = cond(rank(B) == 1, lambda: B[..., None], lambda: B)
    A = transpose(A)
    B = transpose(B)
    return transpose(lstsq(B, A, l2_regularizers=l2_regularizers, fast=fast,
                           rcond=rcond), name=name)


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
        WARNING: default is True (same as numpy // different from tf)
    compute_uv : bool, default=True
        If True, return U, S, V. Else return S.
    hermitian : bool, default=False
        Assume that the input matrices are hermitian.
        WARNING: Only used by the numpy implementation.
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

        WARNING: Order is U, S, V (same as numpy),
                 while tf's order is S, U, V.

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
def factorial(x):
    """Factorial of a number."""
    if is_tensor(x, 'tf'):
        input_dtype = x.dtype
        float_types = (tf.dtypes.float16, tf.dtypes.float32, tf.dtypes.float64)
        if input_dtype not in float_types:
            x = cast(x, 'float32')
        x = tf.where(x == 0,
                     tf.ones_like(x, shape=[1] * rank(x)),
                     tf.math.exp(tf.math.lgamma(x)))
        x = cast(x, input_dtype)
        return x
    else:
        return np.math.factorial(x)


@tensor_compat
def unique(input, return_index=False, return_inverse=False,
           return_counts=False, index_dtype='int32', name=None):
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
    cart : (*arrangements, *shape) tensor or array
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
    perm : (output_length, *shape) tensor or array
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


def expm(input, name=None):
    """Matrix exponential of (a field of) tensors / arrays.

    WARNING: tf supports fields of matrices but np only support pure
             matrices (rank(input) == 2).
    """
    if is_tensor(input, 'tf'):
        return tf.linalg.expm(input, name=name)
    else:
        return sp.linalg.expm(input)


def logm(input, name=None):
    """Matrix logarithm of (a field of) tensors / arrays.

    WARNING: tf supports fields of matrices but np only support pure
             matrices (rank(input) == 2).
    """
    if is_tensor(input, 'tf'):
        return tf.linalg.logm(input, name=name)
    else:
        return sp.linalg.logm(input)
