from rllab.misc.overrides import overrides
from rllab.misc import special2 as special
from rllab.spaces.discrete import Discrete


class OneHot(Discrete):
    """
    The same as Discrete, except all input and output should be one-hot vectors.
    """
    @overrides
    def sample(self):
        return special.to_onehot(
            super().sample(),
            self.n
        )

    @overrides
    def contains(self, x):
        return super().contains(self.from_onehot(x))

    @overrides
    def __repr__(self):
        return "OneHot(%d)" % self.n

    def weighted_sample(self, weights):
        return special.to_onehot(
            self.weighted_sample(weights),
            self.n
        )

    def from_onehot(self, x):
        return special.from_onehot(x)

    def to_onehot(self, x):
        return special.to_onehot(x, self.n)

    @overrides
    def flatten(self, x):
        return x

    @overrides
    def unflatten(self, x):
        return x

    @overrides
    def flatten_n(self, x):
        raise NotImplementedError()

    @overrides
    def unflatten_n(self, x):
        raise NotImplementedError()

    @property
    def default_value(self):
        return special.to_onehot(super().default_value, self.n)

    @overrides
    def __eq__(self, other):
        if not isinstance(other, OneHot):
            return False
        return self.n == other.n
