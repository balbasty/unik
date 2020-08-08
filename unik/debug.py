"""Runtime assertions."""
import tensorflow as tf
import numpy as np

from .magik import tensor_compat


@tensor_compat
def assert_(cond, data=None, message=None, summarize=None, name=None,
            error_class=ValueError):
    """Assert that a condition is true.

    Parameters
    ----------
    cond : bool_like
        Condition to enforce.
    data : list of tensor_like
        Tensors upon which the condition depends.
    message : str, optional
        Message to prepend to the default message.
    summarize : int,
        Print this many entries of each tensor.
    name : str, optional
        Name for this operation.
    error_class : type, default=tf.errors.InvalidArgumentError
        Error to raise on failure.

    Returns
    -------
    If not tf.executing_eagerly() and left or right is a tensor:
        Op that raises `error_class` if `x == y` is False.
        This can be used with `tf.control_dependencies` inside of
        `tf.function`s to block followup computation until the check
        has executed.
    Else:
        None

    Raises
    ------
    If tf.executing_eagerly() or left and right are not tensors:
        error_class : if `x == y` is False

    """
    data = data or []
    data = [d for d in data if tf.is_tensor(d)]
    if not tf.executing_eagerly() and (tf.is_tensor(cond) or len(data) > 0):
        try:
            return tf.debugging.Assert(cond, data, summarize, name)
        except tf.errors.InvalidArgumentError as e:
            raise error_class(message + '\n' + e.message)
    else:
        if not cond:
            message = message or ''
            raise error_class(message)


@tensor_compat
def assert_op(op, *args, **kwargs):
    """Assert that an operation on two tensors is True everywhere.

    Parameters
    ----------
    op : str
        Operation (e.g., 'equal', 'not_equal', 'greater', ...)
    *args : tensor_like
        Tensors on which to apply the operation.

    Other Parameters
    ----------------
    message : str, optional
        Message to prepend to the default message..
    summarize : int,
        Print this many entries of each tensor.
    name : str, optional
        Name for this operation.
    error_class : type, default=tf.errors.InvalidArgumentError
        Error to raise on failure.

    Returns
    -------
    If not tf.executing_eagerly() and any arg is a tensor:
        Op that raises `error_class` if `all(op(*args))` is False.
        This can be used with `tf.control_dependencies` inside of
        `tf.function`s to block followup computation until the check
        has executed.
    Else:
        None

    Raises
    ------
    If tf.executing_eagerly() or left and no arg is a tensor:
        error_class : if `all(op(*args))` is False

    """

    def import_from(module, name):
        module = __import__(module, fromlist=[name])
        return getattr(module, name)

    message = kwargs.pop('message', None) or ''
    message = message + 'Condition `x {} y` did not hold.'.format(op)
    kwargs['message'] = message

    if any((tf.is_tensor(a) for a in args)):
        op = import_from('tensorflow', op)
        return assert_(tf.reduce_all(op(*args)), args, **kwargs)
    else:
        op = import_from('numpy', op)
        return assert_(np.all(op(*args)), args, **kwargs)


def assert_equal(left, right, **kwargs):
    return assert_op('equal', left, right, **kwargs)


def assert_not_equal(left, right, **kwargs):
    return assert_op('not_equal', left, right, **kwargs)


def assert_greater(left, right, **kwargs):
    return assert_op('greater', left, right, **kwargs)


def assert_greater_equal(left, right, **kwargs):
    return assert_op('greater_equal', left, right, **kwargs)


def assert_less(left, right, **kwargs):
    return assert_('less', left, right, **kwargs)


def assert_less_equal(left, right, **kwargs):
    return assert_('less_equal', left, right, **kwargs)


# TODO: more operators
