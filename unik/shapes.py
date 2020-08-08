"""Access / change tensor shape."""
import tensorflow as tf
import numpy as np

from .magik import tensor_compat
from .types import cast, is_tensor, has_tensor, as_tensor, convert_dtype
from .various import name_tensor
from ._cond import cond


@tensor_compat(map_batch=False)
def shape(input, name=None):
    """Shape of a tensor / array."""
    if has_tensor(input, 'tf'):
        true_shape = input.shape
        sym_shape = unstack(tf.shape(input))
        input_shape = [t if t is not None else s
                       for t, s in zip(true_shape, sym_shape)]
        if name is not None:
            input_shape = name_tensor(input_shape, name)
        return input_shape
    else:
        return np.shape(input)


def set_shape(input, shape):
    """Specify the shape of a tensor.

    If the input is not a tf tensor, this function checks that its shape
    matches the specified shape.

    """
    return _set_shape(input, shape)


def _set_shape(input, _shape):
    if is_tensor(input, 'tf'):
        input.set_shape(_shape)
    else:
        input_shape = shape(input)
        if not all(i == o for i, o in zip(input_shape, _shape)
                   if o is not None):
            raise ValueError('Tensor shape and prototype do not match: '
                             '{} and {}'.format(input_shape, _shape))
    return input


@tensor_compat
def size(input, out_type='int32', name=None):
    """Return the number of elements in a tensor.

    Parameters
    ----------
    input : tensor_like
        Input tensor or array.
    out_type : str or type
        Output data type.
    name : str
        Operation name.

    Returns
    -------
    output : () tensor or array
        Number of elements in ``input``.
        Returns a tensor if tf executes eagerly or if the input is a tensor.
        Else, returns a numpy array.

    """
    if is_tensor(input, 'tf'):
        out_type = convert_dtype(out_type, 'tf')
        return tf.size(input, out_type=out_type, name=name)
    else:
        out_type = convert_dtype(out_type, 'np')
        return np.array(np.size(input), dtype=out_type)


@tensor_compat
def rank(input, name=None):
    """Return the rank / number of dimensions of a tensor.

    Parameters
    ----------
    input : tensor_like
        Input tensor or array.
    name : str
        Operation name.

    Returns
    -------
    output : () tensor or int
        Number of elements in ``input``.
        Returns a tensor if tf executes eagerly or if the input is a tensor.
        Else, returns an int.

    """
    return name_tensor(length(shape(as_tensor(input))), name)


def ndim(input, name=None):
    """Alias for `rank`."""
    return rank(input, name)


@tensor_compat
def reshape(input, shape, name=None):
    """Reshape tensor / array."""
    if has_tensor([input, shape], 'tf'):
        input = tf.reshape(input, shape, name=name)
    else:
        input = np.reshape(input, shape)
    return input


@tensor_compat
def flatten(input, name=None):
    """Flatten a tensor / array."""
    return reshape(input, [-1], name=name)


@tensor_compat
def stack(inputs, axis=0, name='stack'):
    """Stack tensors / arrays."""
    if has_tensor(inputs, 'tf'):
        inputs = tf.stack(inputs, axis, name=name)
    else:
        inputs = np.stack(inputs, axis)
    return inputs


@tensor_compat
def unstack(input, num=None, axis=0, name='unstack'):
    """Stack tensors / arrays."""
    if is_tensor(input, 'tf'):
        input = tf.unstack(input, num=num, axis=axis, name=name)
    elif is_tensor(input, 'np'):
        if num is None:
            num = shape(input)[axis]
        input = [np.take(input, i, axis=axis) for i in range(num)]
    else:
        if axis == 0:
            return list(input)
        else:
            raise ValueError('Non-tensor types can only be usntacked '
                             'along the first axis.')
    return input


@tensor_compat
def concat(inputs, axis=0, name='concat'):
    """Concatenate tensors / arrays."""
    if has_tensor(inputs, 'tf'):
        inputs = tf.concat(inputs, axis, name=name)
    elif has_tensor(inputs, 'np'):
        inputs = np.concatenate(inputs, axis)
    elif axis == 0:
        tmp = inputs
        inputs = []
        for i in tmp:
            try:
                i = list(i)
            except TypeError:
                i = [i]
            inputs += list(i)
    else:
        inputs = np.asarray(inputs)
        inputs = concat(inputs, axis, name)
    return inputs


@tensor_compat
def tile(input, multiples, name=None):
    """Tile tensor / array."""
    if tf.is_tensor(input) or tf.is_tensor(multiples):
        input = tf.tile(input, multiples, name=name)
    else:
        input = np.tile(input, multiples)
    return input


@tensor_compat
def transpose(input, axes=None, conjugate=False, name='transpose'):
    """Transpose tensor / array."""
    if tf.is_tensor(input) or tf.is_tensor(axes) or tf.is_tensor(conjugate):
        input = tf.transpose(input, axes, conjugate, name=name)
    else:
        input = np.transpose(input, axes)
        if conjugate:
            input = np.conjugate(input, out=input)
    return input


@tensor_compat
def length(tensor):
    """Length of tensor / array. """
    if is_tensor(tensor):
        return shape(tensor)[0]
    else:
        return len(tensor)


@tensor_compat
def expand_dims(tensor, axis=None, ndim=1):
    """Inserts one or more singleton dimensions.

    Parameters
    ----------
    tensor : tensor_like
        Input tensor.
    axis : int, default=None
        Position where new dimensions are inserted.
    ndim : int, default=1
        Number of singleton dimensions to insert.

    Returns
    -------
    expanded_tensor : tensor or array
        Tensor with more dimensions.

    """
    tensor = as_tensor(tensor)
    ndim0 = length(shape(tensor))
    axis = cond(ndim0 == 0,
                lambda: 0,
                lambda: ((ndim0 + axis) % ndim0) + (axis < 0))
    new_shape = concat(
        (shape(tensor)[:axis], [1] * ndim, shape(tensor)[axis:]))
    new_shape = cast(new_shape, 'int64', keep_none=True)
    tensor = reshape(tensor, new_shape)
    return tensor
