import tensorflow as tf
import numpy as np


def _is_tensor(type):
    """Return a callable checker for the specified type.

    Parameters
    ----------
    type : {'np', 'tf', 'k', 's', None} or type, default=None
        Type of tensor that should be detected. If None, tf and np tensors
        return True.
        * 'np': numpy ndarray
        * 'tf': tensorflow Tensor
        * 'k': keras tensor -> 'tf' + hasattr('_keras_history')
        * 's': symbolic tensor -> 'tf' + hasattr('op')

    Returns
    -------
    check : callable -> bool
        Function that takes a variable and returns True if it is of the
        expected type.

    """

    def is_k(x):
        return tf.is_tensor(x) and hasattr(x, '_keras_history')

    def is_tf(x):
        return tf.is_tensor(x)

    def is_np(x):
        return isinstance(x, np.ndarray)

    if type is None:
        check = lambda x: (is_tf(x) or is_np(x))
    elif type == 'tf':
        check = is_tf
    elif type == 'np':
        check = is_np
    elif type == 'k':
        check = is_k
    else:
        check = lambda x: isinstance(x, type)
    return check


def is_tensor(input, type=None):
    """Check if a variable is a tensor_like.

    Parameters
    ----------
    input : object
    type : {'np', 'tf', 'k', 's', None} or type, default=None
        Type of tensor that should be detected. If None, tf and np tensors
        return True.
        * 'np': numpy ndarray
        * 'tf': tensorflow Tensor
        * 'k': keras tensor -> 'tf' + hasattr('_keras_history')
        * 's': symbolic tensor -> 'tf' + hasattr('op')

    Returns
    -------
    is_tensor : bool

    """
    check = _is_tensor(type)
    return check(input)


def has_tensor(input, type=None, mode='any'):
    """Check if a variable is a tensor_like or nested list of tensor_like.

    Parameters
    ----------
    input : object
    type : {'np', 'tf', 'k', 's', None} or type, default=None
        Type of tensor that should be detected. If None, tf and np tensors
        return True.
        * 'np': numpy ndarray
        * 'tf': tensorflow Tensor
        * 'k': keras tensor -> 'tf' + hasattr('_keras_history')
        * 's': symbolic tensor -> 'tf' + hasattr('op')
    mode : {'any', 'all'} or callable, default='any'
        Aggregation mode.

    Returns
    -------
    has_tensor : bool

    """
    check = _is_tensor(type)
    mode = all if mode == 'all' else any
    if isinstance(input, list) or isinstance(input, tuple):
        return mode(has_tensor(i, type, mode) for i in input)
    if isinstance(input, dict):
        return mode(has_tensor(i, type, mode) for i in input.values())
    return check(input)


def convert_dtype(dtype, otype=None):
    """Convert datatype object to numpy or tensorflow datatype type.

    Parameters
    ----------
    dtype : str or type or np.dtype or tf.dtypes.DType
        Input datatype object.
    otype : {'np', 'tf'} or type
        Output datatype type.

    Returns
    -------
    dtype : np.dtype or tf.dtypes.DType
        Output datatype object.

    """
    if dtype is None:
        return None
    if otype is None:
        return dtype
    elif otype in ('np', np.dtype):
        if isinstance(dtype, tf.dtypes.DType):
            return dtype.as_numpy_dtype
        else:
            return np.dtype(dtype)
    elif otype in ('tf', tf.dtypes.DType):
        return tf.as_dtype(dtype)
