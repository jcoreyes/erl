import abc

from railrl.policies.memory.memory_policy import MemoryPolicy


class RnnCellPolicy(MemoryPolicy, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def rnn_cell(self):
        """
        :return: A TensorFlow RNNCell.
        """
        pass

    @abc.abstractmethod
    def get_init_state_placeholder(self):
        """
        :return: A new tf.placeholder for the initial state of the memory.
        """
        pass

    @property
    @abc.abstractmethod
    def rnn_cell_scope(self):
        """
        Return scope under which self.rnn_cell is made Tensorflow RNNCell.
        :return:
        """
        pass
