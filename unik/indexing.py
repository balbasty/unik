"""Access / change tensor shape."""
import tensorflow as tf
import numpy as np

from .magik import *
from .types import *
from .shape import *
from ._utils import pop


@tensor_compat
def gather(input, indices, validate_indices=None,
           axis=None, batch_dims=0, name=None):
    """Gather / Take values from a tensor / array along an axis."""
    if tf.is_tensor(input) or tf.is_tensor(indices) \
            or tf.is_tensor(axis) or tf.is_tensor(batch_dims):
        return tf.gather(input, indices, validate_indices,
                         axis, batch_dims, name)
    else:
        if batch_dims > 0:
            raise NotImplementedError()
        return np.take(input, indices, axis=axis, mode='raise')


@tensor_compat
def scatter(indices, updates, *args, **kwargs):
    """Scatter `updates` at `indices` into a tensor.

    Signatures
    ----------
    scatter(indices, updates, shape, mode='new', axis=0, name=None)
    scatter(indices, updates, input, mode, axis=0, name=None)

    Parameters
    ----------
    indices - (*ind_shape, L) tensor_like[int]
        ND-indices in which to place the `updates`. The last dimension
        maps to dimensions of the output tensor.

    updates - (*up_shape, *slice_shape) tensor_like or scalar
        Values to place in the tensor.

    shape - vector_like[int], if mode == 'new'
        Shape of the output tensor.

    input - (*shape) tensor_like, if mode != 'new'
        Tensor in which to place `updates`.

    mode - {'new', 'update', 'add', 'sub', 'min', 'max'}, default='new'
        Scatter mode.

    name - str, optional
        A name for the operation.

    Returns
    -------
    output - (*shape) tensor or array
        Tensor with updated values.

    """
    # Parse arguments
    args = list(args)
    kwargs = dict(kwargs)
    mode = pop(args, 1) if len(args) > 1 else kwargs.pop('mode', 'new')
    if mode == 'new':
        input = []
        _shape = pop(args, 0) if len(args) > 0 else kwargs.pop('shape', None)
    else:
        input = pop(args, 0) if len(args) > 0 else kwargs.pop('input', None)
        _shape = shape(input)
    name = pop(args, 0) if len(args) > 0 else kwargs.pop('name', None)

    # Ensure tensors
    if has_tensor([indices, updates, _shape, input], 'tf'):
        updates = as_tensor(updates, 'tf')
        indices = as_tensor(indices, 'tf')
    elif has_tensor([indices, updates, _shape, input], 'np'):
        updates = as_tensor(updates, 'np')
        indices = as_tensor(indices, 'np')
    else:
        updates = as_tensor(updates)
        indices = as_tensor(indices)

    if mode == 'new':
        # Mode new: allocate tensor and populate
        if has_tensor([indices, updates, _shape], 'tf'):
            print(indices.dtype)
            return tf.scatter_nd(indices, updates, _shape, name=name)
        else:
            # np.put works with linear indices only.
            # NOTE: with this implementation, ind_shape and up_shape
            # must be exactly equal, not just broadcastable.
            output = zeros_like(updates, shape=_shape)
            indices = reshape(indices, [-1, shape(indices)[-1]])
            indices = sub2ind(transpose(indices), _shape)
            updates = flatten(updates)
            np.put(output, indices, updates)
            return output
    else:
        if has_tensor([indices, updates, input], 'tf'):
            if mode == 'update':
                scatter_fn = tf.tensor_scatter_nd_update
            elif mode == 'add':
                scatter_fn = tf.tensor_scatter_nd_add
            elif mode == 'sub':
                scatter_fn = tf.tensor_scatter_nd_sub
            elif mode == 'min':
                scatter_fn = tf.tensor_scatter_nd_min
            elif mode == 'max':
                scatter_fn = tf.tensor_scatter_nd_max
            else:
                raise ValueError('Unknown operation {}'.format(mode))
            updates = cast(updates, dtype(input))
            return scatter_fn(input, indices, updates, name=name)
        else:
            # If mode != 'update', equivalent to:
            # 0) the left-hand side is the input tensor
            # 1) generate right-hand side using mode scatter with mode 'new'
            # 2) apply op(LHS, RHS),
            if mode == 'update':
                output = input.copy()
                indices = reshape(indices, [-1, shape(indices)[-1]])
                indices = sub2ind(transpose(indices), _shape)
                updates = flatten(updates)
                np.put(output, indices, updates)
                return output
            elif mode == 'add':
                op = lambda x, y: x + y
            elif mode == 'sub':
                op = lambda x, y: x - y
            elif mode == 'min':
                op = lambda x, y: minimum(x, y)
            elif mode == 'max':
                op = lambda x, y: maximum(x, y)
            else:
                raise ValueError('Unknown operation {}'.format(mode))
            updates = scatter(indices, updates, shape=_shape, mode='new')
            return op(input, updates)


@tensor_compat
def sub2ind(subs, shape):
    """Convert sub indices (i, j, k) into linear indices.

    The rightmost dimension is the most rapidly changing one
    -> if shape == [D, H, W], the strides are therefore [H*W, W, 1]

    Parameters
    ----------
    subs : (D, *shape) tensor_like
        List of sub-indices. The first dimension is the number of dimension.
        Each element should have the same number of elements and shape.
    shape : (D,) vector_like
        Size of each dimension. Its length should be the same as the
        first dimension of ``subs``.

    Returns
    -------
    ind : (*shape) tensor or array
        Linear indices

    """
    *subs, ind = unstack(subs)
    stride = cumprod(shape[1:], reverse=True)
    for i, s in zip(subs, stride):
        ind = ind + as_tensor(i) * s
    return ind


@tensor_compat
def where(cond, x=None, y=None, name=None):
    """Select values from two tensors based on a condition."""
    if has_tensor([cond, x, y], 'tf'):
        return tf.where(cond, x, y, name)
    else:
        if x is None and y is None:
            return np.where(cond)
        else:
            return np.where(cond, x, y)


@tensor_compat
def boolean_mask(input, mask, axis=0, name='boolean_mask'):
    """Gather elements from a tensor / array using a mask."""
    input = as_tensor(input)
    if has_tensor([input, mask], 'tf'):
        return tf.boolean_mask(input, mask, axis=axis, name=name)
    else:
        axis = axis or 0
        slices = (slice(None, None),) * axis + (mask,) + (Ellipsis,)
        return input.__getitem__(slices)
