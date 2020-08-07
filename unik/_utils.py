"""Internal utilities (not exposed to the API)."""


def pop(obj, *args, **kwargs):
    """Safe pop for list or dict.

    Parameters
    ----------
    obj : list or dict
        Input collection.
    elem : default=-1 fot lists, mandatory for dicts
        Element to pop.
    default : optional
        Default value.
        Raise exception if elem does not exist and default not provided.

    Returns
    -------
    value :
        Popped value

    """
    # Parse arguments
    args = list(args)
    kwargs = dict(kwargs)
    elem = None
    elem_known = False
    default = None
    default_known = False
    if len(args) > 0:
        elem = args.pop(0)
        elem_known = True
    if len(args) > 0:
        default = args.pop(0)
        default_known = True
    if len(args) > 0:
        raise ValueError('Maximum two positional arguments. Got {}'
                         .format(len(args)+2))
    if 'elem' in kwargs.keys():
        if elem_known:
            raise ValueError('Argument `elem` cannot be passed as both a '
                             'positional and keyword.')
        else:
            elem = kwargs.pop('elem')
            elem_known = True
    if 'default' in kwargs.keys():
        if default_known:
            raise ValueError('Argument `default` cannot be passed as both a '
                             'positional and keyword.')
        else:
            default = kwargs.pop('default')
            default_known = True

    # --- LIST ---
    if isinstance(obj, list):
        if not elem_known:
            elem = -1
        if default_known:
            try:
                value = obj.pop(elem)
            except IndexError:
                value = default
        else:
            value = obj.pop(elem)

    # --- DICT ---
    elif isinstance(obj, dict):
        if not elem_known:
            raise ValueError('Argument `elem` is mandatory for type dict.')
        if default_known:
            value = obj.pop(elem, default)
        else:
            value = obj.pop(elem)

    # --- OTHER ---
    else:
        raise TypeError('Input object should be a list or dict.')

    return value


def _apply_nested(structure, fn):
    """Recursively apply a function to the elements of a list/tuple/dict."""
    if isinstance(structure, list):
        return list(_apply_nested(elem, fn) for elem in structure)
    elif isinstance(structure, tuple):
        return tuple(_apply_nested(elem, fn) for elem in structure)
    elif isinstance(structure, dict):
        return dict([(k, _apply_nested(v, fn))
                     for k, v in structure.items()])
    else:
        return fn(structure)
