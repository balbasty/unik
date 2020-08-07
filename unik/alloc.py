"""Allocation / Factories."""
import tensorflow as tf
import numpy as np

from .magik import *
from .types import *
from .shape import *
from ._utils import pop


@tensor_compat
def zeros(shape, dtype='float32', name=None):
    """Create a tensor / array filled with zeros."""
    if has_tensor(shape, 'tf'):
        dtype = convert_dtype(dtype, 'tf')
        return tf.zeros(shape, dtype, name)
    else:
        dtype = convert_dtype(dtype, 'np')
        return np.zeros(shape, dtype)


@tensor_compat
def zeros_like(input, shape=None, dtype=None, name=None):
    """Create a tensor / array filled with zeros."""
    if has_tensor([input, shape], 'tf'):
        dtype = convert_dtype(dtype, 'tf')
        return tf.zeros_like(input, shape=shape, dtype=dtype, name=name)
    else:
        dtype = convert_dtype(dtype, 'np')
        return np.zeros_like(input, shape=shape, dtype=dtype)


@tensor_compat
def ones(shape, dtype='float32', name=None):
    """Create a tensor / array filled with ones."""
    if is_tensor(shape, 'tf'):
        dtype = convert_dtype(dtype, 'tf')
        return tf.ones(shape, dtype, name)
    else:
        dtype = convert_dtype(dtype, 'np')
        return np.ones(shape, dtype)


@tensor_compat
def ones_like(input, shape=None, dtype=None, name=None):
    """Create a tensor / array filled with ones."""
    if has_tensor([input, shape], 'tf'):
        dtype = convert_dtype(dtype, 'tf')
        return tf.ones_like(input, shape=shape, dtype=dtype, name=name)
    else:
        dtype = convert_dtype(dtype, 'np')
        return np.ones_like(input, shape=shape, dtype=dtype)


@tensor_compat
def fill(shape, value, dtype=None, name=None):
    """Create a uniformly filled tensor / array."""
    if has_tensor([value, shape], 'tf'):
        value = cast(value, dtype)
        return tf.fill(value, shape, name)
    else:
        return np.full(shape, value, dtype)


@tensor_compat
def fill_like(input, value, shape=None, dtype=None, name=None):
    """Create a uniformly filled tensor / array."""
    input = as_tensor(input)
    dtype = dtype or input.dtype
    if has_tensor([input, value, shape], 'tf'):
        value = cast(value, dtype)
        return tf.fill(value, input.shape, name)
    else:
        dtype = dtype or input.dtype
        if shape is None:
            return np.full_like(input, value, dtype=dtype)
        else:
            return np.full(shape, value, dtype=dtype)


@tensor_compat
def eye(num_rows, num_columns=None, batch_shape=None, dtype='float32',
        name=None):
    """Identity matrix.

    Parameters
    ----------
    num_rows : scalar_like
    num_columns : scalar_like, default=num_rows
    batch_shape : vector_like, default=None
    dtype : str or type, default='float32'
    name : str, optional

    Returns
    -------
    mat : (*batch_shape, R, C) tensor or array

    """
    if has_tensor([num_rows, num_columns, batch_shape], 'tf'):
        mat = tf.eye(num_rows, num_columns, batch_shape, dtype, name)
    else:
        mat = np.eye(num_rows, num_columns, dtype=dtype)
        if batch_shape is not None:
            mat = np.expand_dims(mat, ndim=length(batch_shape))
            mat = tile(mat, batch_shape)
    return mat


@tensor_compat
def one_hot(indices, depth, on_value=None, off_value=None, axis=None,
            dtype=None, name=None):
    if has_tensor([indices, on_value, off_value], 'tf'):
        return tf.one_hot(indices, depth, on_values=on_value,
                          off_values=off_value, dtype=dtype, name=name)
    else:
        if dtype is None and on_value is not None:
            dtype = np.as_tensor(on_value).dtype
        if dtype is None and off_value is not None:
            dtype = np.as_tensor(off_value).dtype
        dtype = convert_dtype(dtype, 'np')
        if on_value is None:
            on_value = np.as_tensor(1).astype(dtype)
        if off_value is None:
            off_value = np.as_tensor(0).astype(dtype)
        output = stack((fill_like(indices, off_value),
                        fill_like(indices, on_value)), axis=axis)
        output[indices, 0] = on_value
        output[indices, 1] = off_value
        # TODO: wrong axis!
        return output


@tensor_compat
def range(*args, **kwargs):
    """Either tf.range or np.arange.

    Parameters
    ----------
    start : default=0
    end
    step : default=1
    dtype : str or type, default=None
    name : str, optional

    Returns
    -------
    range : tensor or array
        Range start:end:step

    """
    args = list(args)
    if any(tf.is_tensor(a) for a in args):
        if len(args) > 3:
            dtype = pop(args, 3)
            dtype = convert_dtype(dtype, 'tf')
            args.insert(3, dtype)
        else:
            dtype = convert_dtype(kwargs.pop('dtype', None), 'tf')
            kwargs['dtype'] = dtype
        return tf.range(*args, **kwargs)
    else:
        if len(args) == 5:
            args = args[:-1]
        kwargs.pop('name', None)
        if len(args) > 3:
            dtype = pop(args, 3)
            dtype = convert_dtype(dtype, 'np')
            args.insert(3, dtype)
        else:
            dtype = convert_dtype(kwargs.pop('dtype', None), 'np')
            kwargs['dtype'] = dtype
        return np.arange(*args, **kwargs)


@tensor_compat
def diag(input, k=0, name='diag'):
    """Extract / set the k-th diagonal of a a matrix."""
    if tf.is_tensor(input):

        def matrix_from_diag(input, k):
            input_shape = (shape(input)[0],) * 2
            zero = tf.zeros_like(input, shape=input_shape)
            return tf.linalg.set_diag(zero, input, k=k)

        return cond(rank(input) >= 1,
                    lambda: tf.linalg.diag(input, k=k),
                    lambda: tf.linalg.tensor_diag(input) if k == 0
                    else matrix_from_diag(input, k),
                    name=name)
    else:
        return np.diag(input, k=k)
