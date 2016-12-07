import abc
import copy
import math
import random


class Hyperparameter(metaclass=abc.ABCMeta):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name


class RandomHyperparameter(Hyperparameter):
    def __init__(self, name):
        super().__init__(name)
        self._last_value = None

    @abc.abstractmethod
    def generate_next_value(self):
        """Return a value for the hyperparameter"""
        return

    def generate(self):
        self._last_value = self.generate_next_value()
        return self._last_value


class EnumParam(RandomHyperparameter):
    def __init__(self, name, possible_values):
        super().__init__(name)
        self.possible_values = possible_values

    def generate_next_value(self):
        return random.choice(self.possible_values)


class LogFloatParam(RandomHyperparameter):
    def __init__(self, name, min_value, max_value):
        super(LogFloatParam, self).__init__(name)
        self._linear_float_param = LinearFloatParam("log_" + name,
                                                    math.log(min_value),
                                                    math.log(max_value))

    def generate_next_value(self):
        return math.e ** (self._linear_float_param.generate())


class LinearFloatParam(RandomHyperparameter):
    def __init__(self, name, min_value, max_value):
        super(LinearFloatParam, self).__init__(name)
        self._min = min_value
        self._delta = max_value - min_value

    def generate_next_value(self):
        return random.random() * self._delta + self._min


class LinearIntParam(RandomHyperparameter):
    def __init__(self, name, min_value, max_value):
        super(LinearIntParam, self).__init__(name)
        self._min = min_value
        self._max = max_value

    def generate_next_value(self):
        return random.randint(self._min, self._max)


class ListedParam(Hyperparameter):
    """
    Represents a list of possible _hyperparameters. Used to do a sweep over a
    fixed list of _hyperparameters.
    """
    def __init__(self, name, values):
        super().__init__(name)
        self._name = name
        self._values = values
        self._last_value = None
        self._i = 0
        self._num_values = len(values)

    @property
    def values(self):
        return self._values

    def generate(self):
        self._last_value = self.values[self._i]
        self._i = (self._i + 1) % self._num_values
        return self._last_value


class FixedParam(ListedParam, RandomHyperparameter):
    def __init__(self, name, value):
        super().__init__(name, [value])
        self._value = value

    def generate_next_value(self):
        return self._value


class RandomHyperparameterSweeper(object):
    def __init__(self, hyperparameters=None):
        self._hyperparameters = hyperparameters or []
        self._validate_hyperparameters()
        self._default_kwargs = {}

    def _validate_hyperparameters(self):
        names = set()
        for hp in self._hyperparameters:
            name = hp.name
            if name in names:
                raise Exception("Hyperparameter '{0}' already added.".format(
                    name))
            names.add(name)

    def set_default_parameters(self, default_kwargs):
        self._default_kwargs = default_kwargs

    def generate_random_hyperparameters(self):
        kwargs = copy.deepcopy(self._default_kwargs)
        for hp in self._hyperparameters:
            kwargs[hp.name] = hp.generate()
        return kwargs

    def sweep_hyperparameters(self, function, num_configs):
        returned_value_and_params = []
        for _ in range(num_configs):
            kwargs = self.generate_random_hyperparameters()
            score = function(**kwargs)
            returned_value_and_params.append((score, kwargs))

        return returned_value_and_params


# TODO(vpong): Finish this or use an implementation online.
class GridHyperparameterSweeper(object):
    """
    Do a grid search over hyperparameters.
    """
    def __init__(self, hyperparameters=None):
        self._hyperparameters = hyperparameters or []
        self._validate_hyperparameters()
        self._default_kwargs = {}
        # names_to_values = []
        # self._hyperparameter_dicts = [dict(
        #
        # )
        #     for params
        # ]

    def _validate_hyperparameters(self):
        names = set()
        for hp in self._hyperparameters:
            assert isinstance(hp, ListedParam)
            name = hp.name
            if name in names:
                raise Exception("Hyperparameter '{0}' already added.".format(
                    name))
            names.add(name)

    def set_default_parameters(self, default_kwargs):
        """
        Set default values for parameters so that when
        iterate_hyperparameters() is called, those parameters will already be
        set.

        :param default_kwargs:
        :return:
        """
        self._default_kwargs = default_kwargs

    def iterate_hyperparameters(self):
        """
        :return: List of dictionaries. Each dictionary is a map from name to
        hyperpameter.
        """
        kwargs = copy.deepcopy(self._default_kwargs)
        for hp in self._hyperparameters:
            kwargs[hp.name] = hp.generate()
        return self._hyperparameter_dicts

    def sweep_hyperparameters(self, function, num_configs):
        returned_value_and_params = []
        for _ in range(num_configs):
            kwargs = self.generate_random_hyperparameters()
            score = function(**kwargs)
            returned_value_and_params.append((score, kwargs))

        return returned_value_and_params
