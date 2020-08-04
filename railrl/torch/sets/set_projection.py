class SetProjection(object):
    def __call__(self, state):
        raise NotImplementedError()


class ProjectOntoAxis(SetProjection):
    def __init__(self, axis_idx_to_value):
        self._axis_idx_to_value = axis_idx_to_value

    def __call__(self, state):
        new_state = state.copy()
        for idx, value in self._axis_idx_to_value.items():
            new_state[idx] = value
        return new_state