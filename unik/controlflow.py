"""Control Flow (conditions / loops / ...)."""
import tensorflow as tf
import numpy as np

from .magik import *
from .types import *
from ._utils import pop

# `cond` is is its own file to break cross-dependencies.
from ._cond import cond


@tensor_compat
def switch_case(index, values, default=None, name='switch_case'):
    """Switch/Case over integer indices.

    Parameters
    ----------
    index : () tensor_like
        Index to switch.
    values : dict(index: callable)
        Pairs of index/function to execute.
    default : callable, optional
        Default function to call if all branches fail.
    name : str, default='switch_case'
        Name for the operation.

    Returns
    -------
    value : tensor or array

    """
    if tf.is_tensor(index):
        return tf.switch_case(index, values, default, name)
    else:
        for key, item in values.items():
            if index == key:
                return item()
        raise IndexError('{} is not a key from `values`.'.format(index))


@tensor_compat
def while_loop(cond, body, loop_vars, shape_invariants=None,
               parallel_iterations=10, back_prop=True, swap_memory=False,
               maximum_iterations=None, name=None):
    """Repeat body while the condition cond is true."""
    if has_tensor(loop_vars, 'tf'):
        return tf.while_loop(cond, body, loop_vars,
                             shape_invariants=shape_invariants,
                             parallel_iterations=parallel_iterations,
                             back_prop=back_prop,
                             maximum_iterations=maximum_iterations,
                             swap_memory=swap_memory,
                             name=name)
    else:
        def _cond_nb_iter(i):
            check = i < maximum_iterations
            i += 1
            return check

        def _cond_nb_iter_alway_true(i):
            return True

        if maximum_iterations is None:
            cond_nb_iter = _cond_nb_iter_alway_true
        else:
            cond_nb_iter = _cond_nb_iter

        i = 0
        while cond_nb_iter(i) and cond(*loop_vars):
            loop_vars = body(*loop_vars)
        return loop_vars


@tensor_compat
def map_fn(fn, elems, dtype=None, parallel_iterations=None, back_prop=True,
           swap_memory=False, infer_shape=True, name=None):
    """Apply a function across the batch dimension of a tensor / array."""

    def _unstack_batch(elems, d):
        # Browse through the input and select the d'th batch element of
        # each np.array
        if isinstance(elems, np.ndarray):
            return elems[d, ...]
        if isinstance(elems, list):
            return list(_unstack_batch(e, d) for e in elems)
        if isinstance(elems, tuple):
            return tuple(_unstack_batch(e, d) for e in elems)
        return elems

    def _stack_batch(batch_elems):
        # Browse through the input and select the d'th batch element of
        # each np.array
        batch_elems = [b for b in batch_elems]
        if len(set(type(b) for b in batch_elems)) > 1:
            raise TypeError('Types not consistent across batches.')
        elems = batch_elems[0]
        if isinstance(elems, np.ndarray):
            return np.stack(batch_elems)
        if isinstance(elems, list):
            return list(_stack_batch(e) for e in batch_elems)
        if isinstance(elems, tuple):
            return tuple(_stack_batch(e) for e in batch_elems)
        return np.stack(elems)

    def _dim_batch(elems):
        # Browse through the input and return the batch dimension
        if isinstance(elems, np.ndarray) and np.ndim(elems) > 0:
            return shape(elems)[0]
        if isinstance(elems, list) or isinstance(elems, tuple):
            dims = set(_dim_batch(e) for e in elems)
            dims = [d for d in dims if d is not None]
            if len(dims) > 1:
                raise ValueError('Batch dimensions are not consistent.')
            if len(dims) == 0:
                return None
            return dims[0]
        return None

    if has_tf_tensor(elems):
        return tf.map_fn(fn, elems, dtype, parallel_iterations, back_prop,
                         swap_memory, infer_shape, name)
    else:
        dim = _dim_batch(elems)
        return _stack_batch(fn(_unstack_batch(elems, d)) for d in range(dim))
