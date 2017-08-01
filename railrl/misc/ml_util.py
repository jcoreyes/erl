"""
General utility functions for machine learning.
"""
import abc


class ScalarSchedule(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_value(self, t):
        pass


class ConstantSchedule(ScalarSchedule):
    def __init__(self, value):
        self._value = value

    def get_value(self, t):
        return self._value


class RampUpSchedule(ScalarSchedule):
    def __init__(
            self,
            min_value,
            max_value,
            ramp_duration,
    ):
        self._min_value = min_value
        self._max_value = max_value
        self._ramp_duration = ramp_duration

    def get_value(self, t):
        return (
            self._min_value
            + (self._max_value - self._min_value)
            * min(1.0, t * 1.0 / self._ramp_duration)
        )

