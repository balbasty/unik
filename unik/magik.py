"""This module implements decorators and helpers to make generic functions.

Unik defines function that work on numpy arrays, tensorflow eager tensors,
tensorflow static tensors and keras tensors. This module defines helpers
that help reaching this goal.

"""
import types
import inspect
import functools
import numpy as np
import tensorflow as tf

from ._utils import pop, _apply_nested
from ._tensor_types import is_tensor, has_tensor, convert_dtype


def _get_keras_from_tensor(x):
    """Return 'keras' or 'tensorflow' depending on which keras
    implementation was used to initialize `x`."""
    if not is_tensor(x, 'k'):
        raise TypeError('Expected Keras tensor '
                        '(with attribute `_keras_history). '
                        'Got a {}.'.format(type(x)))
    layer, _, _ = x._keras_history
    module = layer.__module__.split('.')[0]
    if module not in ('keras', 'tensorflow'):
        raise ValueError('Unknown keras module: {}'.format(module))
    return module


class Value:
    """Compute a value based on the arguments of a function.

    This object (along with `Argument`) allows functions of arguments
    to be defined symbolically, that is, before the values of the
    arguments are known. Their evaluation is deferred until the arguments
    are known.
    """
    def __init__(self, value_fn):
        self.value_fn = value_fn

    def __call__(self, func, args, kwargs):
        return self.value_fn(func, args, kwargs)


class Argument(Value):
    """Symbolic reference to the argument of a function.

    It will be evaluated at run time.
    """

    def _value_fn(self, func, args, kwargs):
        """Generic `__call__` method for `Argument`s."""
        arg_value = get_argument_value(self.name, func, args, kwargs)
        if self.default_set and arg_value is self.default_trigger:
            if isinstance(self.default, Value):
                arg_value = self.default(func, args, kwargs)
            else:
                arg_value = self.default
        return arg_value

    def __init__(self, name, *args, **kwargs):
        """

        Parameters
        ----------
        name : str
            Argument name.
        default : DefaultValue, optional
            Default value.
        default_trigger : default=None
            Returned value that triggers the evaluation of `default`.

        """
        super().__init__(self._value_fn)
        self.name = name
        self.default_set = len(args) > 0 or ('default' in kwargs.keys())
        if self.default_set:
            self.default = args[0] if len(args) > 0 else kwargs['default']
            self.default_trigger = args[1] if len(args) > 1 \
                else kwargs.get('default', None)
        else:
            self.default = None


def get_argument_value(arg_name, func, args=None, kwargs=None,
                       return_default=False):
    """Return the runtime value of the argument of a function."""
    args = [] if args is None else args
    kwargs = {} if kwargs is None else kwargs
    arg_found = False

    # if argument passed as keyword, get its value
    if arg_name in kwargs.keys():
        arg_value = kwargs[arg_name]
        if not return_default:
            return arg_value
        arg_found = True

    # else, we need to inspect the function's signature
    sig = inspect.signature(func)

    # if argument is passed as positional, get its value
    if not arg_found:
        try:
            arg_index = list(sig.parameters).index(arg_name)
        except ValueError:
            raise ValueError('`{}` is not an argument of function `{}`'
                             .format(arg_name, func.__name__))
        if len(args) > arg_index:
            arg_value = args[arg_index]
            if not return_default:
                return arg_value
            arg_found = True

    # else, we return the default value
    arg_default = sig.parameters[arg_name].default
    if not arg_found:
        arg_value = arg_default

    if return_default:
        return arg_value, arg_default
    else:
        return arg_value


def tensor_compat(map_batch=True, return_dtype=None):
    """Decorator: wrap a function in a keras layer when needed.

    If some of the function's arguments are symbolic tensor, but we are
    executing in an eager (i.e., dynamic) context, it means that we
    are effectively building a keras model. The function must therefore
    be wrapped in a lambda layer, to avoid executing it eagerly.

    Parameters
    ----------
    map_batch : bool, default=True
        If True, when some inputs are Keras tensors, the underlying
        function is mapped to each element of the batch using tf.map_fn

    return_dtype : (nested structure of) object or Value, optional
        Datatype of the tensor(s) returned by the function. Can be a
        `Value`, in which case it is evaluated at call time.
        The nested structure of `return_dtype` should be the same as
        that returned by the function. That is, if the function returns
        a tuple of three tensors, return_dtype should have the form:
        `(dtype1, dtype2, dtype3)`

    Returns
    -------
    decorator

    """

    # Catch "wrong" use of the meta-decorator
    # (that is, it hasn't been called/built)
    # In this case, we want the meta-decorator to be equivalent to the
    # decorator. We detect such "wrong" cases by checking if the input
    # is callable.
    if callable(map_batch):
        func = map_batch
        true_decorator = tensor_compat()
        return true_decorator(func)

    map_batch = map_batch
    return_dtype = return_dtype

    def decorator(func):
        """The actual decorator. Its implementation depends on `map_batch`."""

        class ExtractedTensor:
            """Small object to temporarily replace keras tensors."""

            def __init__(self, i):
                self.i = i

            def __call__(self, list_tf):
                return list_tf[self.i]

        def extract_tf(args, kwargs):
            """Browse through a nested object and extract all keras tensors.
            Tensors are stored in a list and replaced by their index in the list."""

            def sub(obj, list_tf):
                if isinstance(obj, list):
                    return list(sub(o, list_tf) for o in obj)
                elif isinstance(obj, tuple):
                    return tuple(sub(o, list_tf) for o in obj)
                elif isinstance(obj, dict):
                    return dict((k, sub(v, list_tf)) for k, v in obj.items())
                elif is_tensor(obj, 'k'):
                    list_tf.append(obj)
                    i = len(list_tf) - 1
                    return ExtractedTensor(i)
                return obj

            list_tf = []
            args = sub(args, list_tf)
            kwargs = sub(kwargs, list_tf)
            return args, kwargs, list_tf

        def insert_tf(args, kwargs, list_tf):
            """Re-insert keras tensors into the original nested object."""

            def sub(obj, list_tf):
                if isinstance(obj, list):
                    return list(sub(o, list_tf) for o in obj)
                elif isinstance(obj, tuple):
                    return tuple(sub(o, list_tf) for o in obj)
                elif isinstance(obj, dict):
                    return dict((k, sub(v, list_tf)) for k, v in obj.items())
                elif isinstance(obj, ExtractedTensor):
                    return obj(list_tf)
                return obj

            args = sub(args, list_tf)
            kwargs = sub(kwargs, list_tf)
            return args, kwargs

        def unpack_args(inp, args, kwargs, return_dtype=None):
            """Calls the original function with the appropriate arguments."""
            # print(func.__name__, tf.executing_eagerly())
            if not isinstance(inp, (list, tuple)):
                inp = [inp]
            args, kwargs = insert_tf(args, kwargs, inp)
            return func(*args, **kwargs)

        def map_unpack_args(inp, args, kwargs, return_dtype):
            """Split tensors along the batch dimension and apply the function
            to each batch element."""
            return_dtype = convert_dtype(return_dtype, 'tf')
            return tf.map_fn(lambda x: unpack_args(x, args, kwargs), inp,
                             dtype=return_dtype)

        if not map_batch:
            map_unpack_args = unpack_args

        @functools.wraps(func)
        def wrap_in_lambda(*args, **kwargs):
            """If the input contains Keras tensors and we're not inside
            the keras graph, wrap the function in a Lambda layer."""

            # Evaluate return_dtype if it is a symbolic value.
            # It must be done before extracting keras tensors.
            def evaluate(value):
                if isinstance(value, Value):
                    return value(func, args, kwargs)
                else:
                    return value
            _return_dtype = _apply_nested(return_dtype, evaluate)

            # print(func.__name__, tf.executing_eagerly(), has_tensor([args, kwargs], 'k'))
            if tf.executing_eagerly() and has_tensor([args, kwargs], 'k'):
                # Split tensor and non-tensor arguments
                args, kwargs, inp = extract_tf(args, kwargs)

                # Select apropriate keras module (internal or external)
                which_keras = _get_keras_from_tensor(inp[0])
                if which_keras.lower() == 'keras':
                    from keras.layers import Lambda
                else:
                    from tensorflow.keras.layers import Lambda

                # There seem to be a bug in keras when the input is a
                # single tensor in a list (the shape is unpacked, but
                # not the dtype, leading to errors).
                if len(inp) == 1:
                    inp = inp[0]

                # Create Lambda layer
                def lambda_fn(x):
                    return map_unpack_args(x, args, kwargs, _return_dtype)
                name = kwargs.get('name') or func.__name__
                return Lambda(lambda_fn, name=name)(inp)
            else:
                return func(*args, **kwargs)

        return wrap_in_lambda

    return decorator





