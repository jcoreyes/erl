from rllab.misc import tensor_utils
import numpy as np


class Path(dict):
    """
    Represents a sequence of states, actions, rewards, terminals, etc.
    """

    def __init__(self):
        super().__init__()
        self._path_length = 0

    def add_all(self, increment_path_length=True, **key_to_value):
        for k, v in key_to_value.items():
            if k not in self:
                self[k] = [v]
            else:
                self[k].append(v)
        if increment_path_length:
            self._path_length += 1

    def get_all_stacked(self):
        output_dict = dict()
        for k, v in self.items():
            output_dict[k] = stack_list(v)
        return output_dict

    def __len__(self):
        return self._path_length


def stack_list(lst):
    if isinstance(lst[0], dict):
        # return tensor_utils.stack_tensor_dict_list(lst)
        return np.array(lst)
    else:
        return tensor_utils.stack_tensor_list(lst)


