import tensorflow as tf
from .magik import tensor_compat


@tensor_compat(map_batch=False)
def name_tensor(input, name):
    """Name a tensor.

    If the input is not a tensor, it is not named.

    Parameters
    ----------
    input : tensor_like
    name : str

    Returns
    -------
    output : tensor_like

    """
    if tf.is_tensor(input):
        return tf.identity(input, name=name)
    else:
        return input
