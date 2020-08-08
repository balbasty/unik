"""Conversion between tensor types / data types."""
import tensorflow as tf
import numpy as np

from .magik import tensor_compat
from .various import name_tensor
from ._utils import _apply_nested


# These are defined in a different file to avoid cross-dependencies
from ._tensor_types import is_tensor, has_tensor, convert_dtype


def has_tf_tensor(input):
    """Check if a variable is a `tf.tensor` or nested list of `tf.tensor`s."""
    return has_tensor(input, 'tf')


def dtype(input):
    """Data type."""
    return as_tensor(input).dtype


def as_tensor(input, force=None, dtype=None, name=None):
    """Convert to a dynamic array/tensor type if possible.

    If tf.executing_eagerly(), return a tensor.
    If the input is a tensor, return a tensor.
    Else, return a numpy array.

    Parameters
    ----------
    input : tensor_like
    force : {'tf', 'np'}, optional
        If 'tf': converts to tf.tensor even if tf.executing_eagerly()
            and the input is a not a tensor. Values inside the tensor
            are then "lost" statically.
        If 'np': converts to np.array even if not tf.executing_eagerly()
            and the input is a tensor. This may imply the creation of a
            session and the evaluation of the computational graph.
        If None:
            If the input is already a np or tf tensor: keep it as is.
            Else if tf.executing_eagerly(): convert to tf.
            Else: convert to np.
    dtype : str or type, optional
        Ensure that the returned tensor has a given data type.
        If None, keep the input datatype (or the default behavior of
        the conversion function).
    name : str, optional
        A name for the output tensor.

    Returns
    -------
    output : tensor or array

    """
    if force == 'tf':
        input = tf.convert_to_tensor(input)
    elif force == 'np':
        if not tf.executing_eagerly() and tf.is_tensor(input):
            input = input.eval(session=tf.compat.v1.Session())
        else:
            input = np.asarray(input)
    elif force is None:
        if is_tensor(input):
            pass
        elif tf.executing_eagerly() or tf.is_tensor(input):
            input = tf.convert_to_tensor(input)
        else:
            input = np.asarray(input)
    else:
        raise ValueError('force should be in (''np'', ''tf'', None). '
                         'Got {}.'.format(force))
    # Postprocessing
    if dtype is not None:
        input = cast(input, dtype=dtype)
    if name is not None:
        input = name_tensor(input, name)
    return input


@tensor_compat(map_batch=False)
def cast(input, dtype, keep_none=False, name=None):
    """Casts a tensor to a new type.

    Parameters
    ----------
    input : iterable or tensor_like
        Input tensor or nested iterable.
    dtype : str or type
        output data type.
    keep_none : bool, default=False
        If True, do not convert None values.
    name : str, optional
        Name for the operation.

    Returns
    -------
    iterable or tensor or array
        Converted object.

    """
    if is_tensor(input, 'tf'):
        dtype = convert_dtype(dtype, 'tf')
        input = tf.cast(input, dtype, name)
    elif is_tensor(input, 'np'):
        dtype = convert_dtype(dtype, 'np')
        input = input.astype(dtype)
    elif isinstance(input, (list, tuple, dict)):
        input = _apply_nested(input,
                              lambda x: cast(x, dtype, keep_none=keep_none))
    elif input is None and keep_none:
        pass
    else:
        dtype = convert_dtype(dtype, 'np')
        input = np.asarray(input).astype(dtype).item()
    return input
