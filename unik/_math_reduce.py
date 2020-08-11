"""Factories for reductions."""
import operator as op
import tensorflow as tf
import numpy as np

from .magik import tensor_compat
from .types import has_tensor, cast
from .shapes import size, shape
from .various import name_tensor
from ._math_for_indexing import sqrt


# --- FACTORY ----------------------------------------------------------


def _build_reduction(tf_op, np_op, py_op=None, docline='', use_dtype=True):
    """Build a unik reduction operator.

    Parameters
    ----------
    tf_op : callable
        Tensorflow implementation of the reduction.
    np_op : callable
        Numpy implementation of the reduction
    py_op : callable
        Python operator used in the reduction.
    docline : str, optional
        First line of the docstring.
    use_dtype : bool, default=True
        Allow the user to choose accumulator dtype.

    Returns
    -------
    reduce : callable
        A reduction function that runs on both tensors and arrays.

    """

    doc = """ across an axis.


        Parameters
        ----------
        input : tensor_like
            Input tensor.

        axis : int or None, default=None
            Axis along which to reduce. If None, flatten the tensor.
            Cannot be a `tf.Tensor`.
        """

    if use_dtype:
        doc = doc + """
        dtype : str or type, default=dtype(input)
            Type of the accumulator and returned array.

            .. np:: if the input has an integer type of less precision
                    than the platform precision, the platform integer
                    (or its unsigned version) is used.
        """

    doc = doc + """
        keepdims : bool, default=False
            If True, the reduced dimension is removed.
            Else, it is kept as a singleton dimension.
        """

    if py_op is not None:
        doc = doc + """
        initial : () tensor_like, optional
            Starting value for the reduction.

            .. tf:: this option is not natively supported, so the initial
                    value is added at the end. This make the overflow
                    behaviour differ between np and tf.
        """

    doc = doc + """
        name : str, optional
            A name for the operation.

        Returns
        -------
        reduced_tensor : tensor or array

        """

    if use_dtype and py_op is not None:
        @tensor_compat
        def reduce(input, axis=None, dtype=None, keepdims=False,
                   initial=None, name=None):
            if has_tensor([input, initial], 'tf'):
                input = input if dtype is None else cast(input, dtype)
                input = tf_op(input, axis, keepdims=keepdims)
                if initial is not None:
                    input = py_op(input, initial)
                return name_tensor(input, name)
            else:
                return np_op(input, axis, dtype=dtype,
                             keepdims=keepdims, initial=initial)
    elif use_dtype:
        @tensor_compat
        def reduce(input, axis=None, dtype=None, keepdims=False, name=None):
            if has_tensor(input, 'tf'):
                input = input if dtype is None else cast(input, dtype)
                input = tf_op(input, axis, keepdims=keepdims)
                return name_tensor(input, name)
            else:
                return np_op(input, axis, dtype=dtype, keepdims=keepdims)
    elif py_op is not None:
        @tensor_compat
        def reduce(input, axis=None, keepdims=False, initial=None, name=None):
            if has_tensor([input, initial], 'tf'):
                input = tf_op(input, axis, keepdims=keepdims)
                if initial is not None:
                    input = py_op(input, initial)
                return name_tensor(input, name)
            else:
                return np_op(input, axis, keepdims=keepdims, initial=initial)
    else:
        @tensor_compat
        def reduce(input, axis=None, keepdims=False, name=None):
            if has_tensor(input, 'tf'):
                input = tf_op(input, axis, keepdims=keepdims)
                return name_tensor(input, name)
            else:
                return np_op(input, axis, keepdims=keepdims)

    reduce.__doc__ = docline + doc
    return reduce


# --- BUILD FROM FACTORY -----------------------------------------------


tfm = tf.math
sum = _build_reduction(tfm.reduce_sum, np.sum, op.add, 'Sum')
prod = _build_reduction(tfm.reduce_prod, np.prod, op.mul, 'Product')
all = _build_reduction(tfm.reduce_all, np.all, op.and_, '"Logical and"')
any = _build_reduction(tfm.reduce_any, np.any, op.or_, '"Logical or"')
max = _build_reduction(tfm.reduce_max, np.max, docline='Maximum', use_dtype=False)
min = _build_reduction(tfm.reduce_min, np.min, docline='Minimum', use_dtype=False)
mean = _build_reduction(tfm.reduce_mean, np.mean, docline='Mean')


# --- SPECIFIC IMPLEMENTATIONS -----------------------------------------
# These reductions cannot be easily implemented using the factory.


@tensor_compat
def std(input, axis=None, dtype=None, ddof=0, keepdims=False, name=None):
    """Standard deviation across an axis.

    Parameters
    ----------
    input : tensor_like
        Input tensor.

    axis : int or None, default=None
        Axis along which to reduce. If None, flatten the tensor.
        Cannot be a `tf.Tensor`.

    dtype : str or type, default=dtype(input)
        Type of the accumulator and returned array.

        .. np:: if the input has an integer type of less precision
                than the platform precision, the platform integer
                (or its unsigned version) is used.

    ddof : int, default=0
        "Delta Degrees of Freedom": the divisor used in the calculation is
        ``N - ddof``, where ``N`` represents the number of elements.

    keepdims : bool, default=False
        If True, the reduced dimension is removed.
        Else, it is kept as a singleton dimension.

    name : str, optional
        A name for the operation.

    Returns
    -------
    reduced_tensor : tensor or array

    """
    if has_tensor(input, 'tf'):
        input = input if dtype is None else cast(input, dtype)
        nb_elem = shape(input)[axis] if axis is not None else size(input)
        input = tf.math.reduce_std(input, axis, keepdims=keepdims)
        if ddof > 0:
            input = input * sqrt(nb_elem / (nb_elem - 1))
        return name_tensor(input, name)
    else:
        return std(input, axis, dtype=dtype, ddof=ddof, keepdims=keepdims)


@tensor_compat
def var(input, axis=None, dtype=None, ddof=0, keepdims=False, name=None):
    """Variance across an axis.

    Parameters
    ----------
    input : tensor_like
        Input tensor.

    axis : int or None, default=None
        Axis along which to reduce. If None, flatten the tensor.
        Cannot be a `tf.Tensor`.

    dtype : str or type, default=dtype(input)
        Type of the accumulator and returned array.

        .. np:: if the input has an integer type of less precision
                than the platform precision, the platform integer
                (or its unsigned version) is used.

    ddof : int, default=0
        "Delta Degrees of Freedom": the divisor used in the calculation is
        ``N - ddof``, where ``N`` represents the number of elements.

    keepdims : bool, default=False
        If True, the reduced dimension is removed.
        Else, it is kept as a singleton dimension.

    name : str, optional
        A name for the operation.

    Returns
    -------
    reduced_tensor : tensor or array

    """
    if has_tensor(input, 'tf'):
        input = input if dtype is None else cast(input, dtype)
        nb_elem = shape(input)[axis] if axis is not None else size(input)
        input = tf.math.reduce_variance(input, axis, keepdims=keepdims)
        if ddof > 0:
            input = input * nb_elem / (nb_elem - 1)
        return name_tensor(input, name)
    else:
        return var(input, axis, dtype=dtype, ddof=ddof, keepdims=keepdims)


@tensor_compat
def cumsum(input, axis=None, dtype=None, exclusive=False, reverse=False,
           name=None):
    """Cumulative sum across an axis."""
    if has_tensor(input, 'tf'):
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
            input = np.take(input, range(shape(input)[axis] - 1),
                            axis=axis)
        input = np.cumsum(input, axis, dtype=dtype)
        if exclusive:
            pad = np.zeros((input.dim, 2), dtype='int')
            pad[axis, 0] = 1
            input = np.pad(input, pad, constant_values=1)
        if reverse:
            input = np.flip(input, axis)
        return input
