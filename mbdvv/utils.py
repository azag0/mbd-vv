from itertools import islice


def last(obj):
    if not isinstance(obj, list):
        return obj
    assert len(obj) == 2
    return obj[-1]


def listify(obj):
    if isinstance(obj, list):
        return obj
    return [obj]


def chunks(iterable, n):
    iterable = iter(iterable)
    while True:
        chunk = list(islice(iterable, n))
        if not chunk:
            break
        yield chunk
