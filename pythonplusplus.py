"""
General purpose Python functions.
"""


# TODO(vpong): probably move this to its own module, not under railrl
def identity(x):
    return x


def clip_magnitude(value, magnitude):
    """
    Clip the magnitude of value to be within some value.

    :param value:
    :param magnitude:
    :return:
    """
    return min(max(value, -magnitude), magnitude)
