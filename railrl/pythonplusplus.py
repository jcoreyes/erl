"""
General purpose Python functions.
"""
import random
from collections import Iterable


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


def are_values_close(value, target, epsilon=1e-3):
    return abs(value - target) <= epsilon


def sample_with_replacement(iterable, num_samples):
    return [random.choice(iterable) for _ in range(num_samples)]


# TODO(vpong): test this
def map_recursive(fctn, x_or_iterable):
    """
    Apply `fctn` to each element in x_or_iterable.

    This is a generalization of the map function since this will work
    recursively for iterables.

    :param fctn: Function from element of iterable to something.
    :param x_or_iterable: An element or an Iterable of an element.
    :return: The same (potentially recursive) iterable but with
    all the elements transformed by fctn.
    """
    # if isinstance(x_or_iterable, Iterable):
    if isinstance(x_or_iterable, list) or isinstance(x_or_iterable, tuple):
        return type(x_or_iterable)(
            map_recursive(fctn, item) for item in x_or_iterable
        )
    else:
        return fctn(x_or_iterable)


def filter_recursive(x_or_iterable):
    """
    Filter out elements that are Falsy (where bool(x) is False) from
    potentially recursive lists.

    :param x_or_iterable: An element or a list.
    :return: If x_or_iterable is not an Iterable, then return x_or_iterable.
    Otherwise, return a filtered version of x_or_iterable.
    """
    if isinstance(x_or_iterable, list):
        new_items = []
        for sub_elem in x_or_iterable:
            filtered_sub_elem = filter_recursive(sub_elem)
            if filtered_sub_elem is not None and not (
                isinstance(filtered_sub_elem, list) and
                len(filtered_sub_elem) == 0
            ):
                new_items.append(filtered_sub_elem)
        return new_items
    else:
        return x_or_iterable
