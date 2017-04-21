"""
General purpose Python functions.
"""
import random
import sys


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


class _Logger(object):
    def __init__(self):
        self.n_chars = 0

    def print_over(self, string):
        """
        Remove anything printed in the last printover call. Then print `string`
        """
        sys.stdout.write("\r" * self.n_chars)
        sys.stdout.write(string)
        sys.stdout.flush()
        self.n_chars = len(string)

    def newline(self):
        sys.stdout.write("\n")
        sys.stdout.flush()
        self.n_chars = 0


line_logger = _Logger()


def dot_map_dict_to_nested_dict(dot_map_dict):
    """
    Convert something like
    ```
    {
        'one.two.three.four': 4,
        'one.six.seven.eight': None,
        'five.nine.ten': 10,
        'five.zero': 'foo',
    }
    ```
    into its corresponding nested dict.

    http://stackoverflow.com/questions/16547643/convert-a-list-of-delimited-strings-to-a-tree-nested-dict-using-python
    :param dot_map_dict:
    :return:
    """
    tree = {}

    for key, item in dot_map_dict.items():
        split_keys = key.split('.')
        if len(split_keys) == 1:
            tree[key] = item
        else:
            t = tree
            for sub_key in split_keys[:-1]:
                t = t.setdefault(sub_key, {})
            last_key = split_keys[-1]
            t[last_key] = item
    return tree


def merge_recursive_dicts(a, b, path=None,
                          ignore_duplicat_keys_in_second_dict=False):
    """
    Merge two dicts that may have nested dicts.
    """
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_recursive_dicts(a[key], b[key], path + [str(key)],
                                      ignore_duplicat_keys_in_second_dict=ignore_duplicat_keys_in_second_dict)
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                duplicate_key = '.'.join(path + [str(key)])
                if ignore_duplicat_keys_in_second_dict:
                    print("duplicate key ignored: {}".format(duplicate_key))
                else:
                    raise Exception(
                        'Duplicate keys at {}'.format(duplicate_key)
                    )
        else:
            a[key] = b[key]
    return a