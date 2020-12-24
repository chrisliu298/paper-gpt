def dict_to_obj(d):
    """Convert a dictionary to a class object"""
    if isinstance(d, list):
        d = [dict_to_obj(x) for x in d]
    if not isinstance(d, dict):
        return d

    class Class:
        pass

    obj = Class()
    for k in d:
        obj.__dict__[k] = dict_to_obj(d[k])
    return obj