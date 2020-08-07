from .types import dtype
from .magik import Value


def dtype_like(arg):
    def implementation(func, args, kwargs):
        if isinstance(arg, Value):
            return dtype(arg(func, args, kwargs))
        else:
            return dtype(arg)

    return Value(implementation)
