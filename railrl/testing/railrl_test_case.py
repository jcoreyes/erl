from railrl.testing.tf_test_case import TFTestCase
from railrl.launchers.launcher_util import reset_execution_environment
from rllab.misc import logger


class RailRLTestCase(TFTestCase):
    """
    RailRL test case ensures that the execution environment is ready for
    launching a railrl algorithm.
    """
    def setUp(self):
        reset_execution_environment()
        logger.set_snapshot_dir("/tmp/railrl-snapshot")
        super().setUp()
