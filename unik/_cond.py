import tensorflow as tf

from .magik import *


@tensor_compat
def cond(pred, true_fn=None, false_fn=None, name=None):
    """Return `true_fn()` if the predicate `pred` is true else `false_fn()`."""
    if tf.is_tensor(pred):
        return tf.cond(pred, true_fn=true_fn, false_fn=false_fn, name=name)
    else:
        if pred:
            if true_fn is not None:
                return true_fn()
        else:
            if false_fn is not None:
                return false_fn()
    return None
