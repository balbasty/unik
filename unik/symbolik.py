import numpy as np
import tensorflow as tf

from .magik import symbolik_helper, Val, Arg
from .types import dtype as _dtype, \
                   result_dtype as _result_dtype


__all__ = ['Val', 'Arg', 'dtype', 'result_dtype']

dtype = symbolik_helper(_dtype)
result_dtype = symbolik_helper(_result_dtype)
