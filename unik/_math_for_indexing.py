import tensorflow as tf
import numpy as np

from .magik import tensor_compat
from .types import has_tensor, is_tensor, cast
from .shapes import shape


@tensor_compat
def cumprod(input, axis=None, dtype=None, exclusive=False, reverse=False,
            name=None):
    """Cumulative product across an axis."""
    if has_tensor(input, 'tf'):
        input = cast(input, dtype)
        if axis is None:
            input = tf.reshape(input, [-1])
            axis = 0
        return tf.math.cumprod(input, axis, exclusive=exclusive,
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
        input = np.cumprod(input, axis, dtype=dtype)
        if exclusive:
            pad = np.zeros((input.ndim, 2), dtype='int')
            pad[axis, 0] = 1
            input = np.pad(input, pad, constant_values=1)
        if reverse:
            input = np.flip(input, axis)
        return input


@tensor_compat
def minimum(x, y, name=None):
    """Minimum of two tensors / arrays."""
    if has_tensor([x, y], 'tf'):
        return tf.math.minimum(x, y, name)
    else:
        return np.minimum(x, y)


@tensor_compat
def maximum(x, y, name=None):
    """Maximum of two tensors / arrays."""
    if has_tensor([x, y], 'tf'):
        return tf.math.maximum(x, y, name)
    else:
        return np.maximum(x, y)


@tensor_compat(map_batch=False)
def sqrt(input, name=None):
    """Element-wise square-root of a tensor."""
    if is_tensor(input, 'tf'):
        return tf.math.sqrt(input, name=name)
    else:
        return np.sqrt(input)
