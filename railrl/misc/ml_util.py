"""
General utility functions for machine learning.
"""
import abc
import math
from collections import deque
import numpy as np


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
    """
    Ramp up linearly and then stop at a max value.
    """
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


class IntRampUpSchedule(RampUpSchedule):
    """
    Same as RampUpSchedule but round output to an int
    """
    def get_value(self, t):
        return int(super().get_value(t))


# TODO(vitchyr)
class PiecewiseLinearSchedule(ScalarSchedule):
    """
    Given a list of (x, t) value-time pairs, return value x at time t,
    and linearly interpolate between the two
    """
    pass


def none_to_infty(bounds):
    if bounds is None:
        bounds = -math.inf, math.inf
    lb, ub = bounds
    if lb is None:
        lb = -math.inf
    if ub is None:
        ub = math.inf
    return lb, ub


class StatConditionalSchedule(ScalarSchedule):
    """
    Every time a (running average of the) statistic dips is above a threshold,
    add `delta` to the outputted value.

    If the statistic is below a threshold, subtract `delta` to the
    outputted value.
    """
    def __init__(
            self,
            init_value,
            stat_bounds,
            running_average_length,
            delta=1,
            value_bounds=None,
            statistic_name=None,
            min_num_stats=None,
    ):
        """
        :param init_value: Initial value outputted
        :param stat_bounds: (min, max) values for the stat. When the running
        average of the stat exceeds this threshold, the outputted value changes.
        :param running_average_length: How many stat values to average. Not
        updates occur until this many samples are taken.
        :param statistic_name: Name of the statistic to follow.
        :param delta: How much to add to the output value when the statistic
        is above the threshold.
        :param value_bounds: (min, max) ints for the value. If None, then the
        outputted value can grow and shrink arbitrarily.
        """
        if min_num_stats is None:
            min_num_stats = running_average_length
        if value_bounds is None:
            value_bounds = -math.inf, math.inf
        value_lb, value_ub = none_to_infty(value_bounds)
        stat_lb, stat_ub = stat_bounds

        assert min_num_stats <= running_average_length
        assert stat_lb < stat_ub
        assert value_lb < value_ub

        self._value = init_value
        self.stat_lb, self.stat_ub = stat_lb, stat_ub
        self._stats = deque(maxlen=running_average_length)
        self.delta = delta
        self.value_lb, self.value_ub = value_lb, value_ub
        self.statistic_name = statistic_name
        self.min_number_stats = min_num_stats

    def get_value(self, t):
        return self._value

    def update(self, stat):
        self._stats.append(stat)
        if len(self._stats) < self.min_number_stats:
            return

        mean = np.mean(self._stats)
        if mean > self.stat_ub:
            self._value += self.delta
        if mean < self.stat_lb:
            self._value -= self.delta

        if self.value_ub is not None:
            self._value = min(self.value_ub, max(self.value_lb, self._value))
