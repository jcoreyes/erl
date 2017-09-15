import joblib
import numpy as np
import torch

from railrl.launchers.launcher_util import run_experiment
from railrl.policies.base import Policy, SerializablePolicy
from railrl.samplers.util import rollout
from railrl.torch.naf import NafPolicy
from rllab.core.serializable import Serializable
from rllab.envs.normalized_env import normalize
from rllab.misc import logger

column_to_path = dict(
    left=(
        '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-14-pusher-3dof-reacher-naf-yolo-left/09-14_pusher-3dof-reacher-naf-yolo_left_2017_09_14_17_52_45_0010/params.pkl'
    ),
    right=(
        '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-14-pusher-3dof-reacher-naf-yolo-right/09-14_pusher-3dof-reacher-naf-yolo_right_2017_09_14_17_52_45_0016/params.pkl'
    ),
    middle=(
        '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-14-pusher-3dof-reacher-naf-yolo-middle/09-14_pusher-3dof-reacher-naf-yolo_middle_2017_09_14_17_52_45_0013/params.pkl'
    ),
)
bottom_path = (
    '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-14-pusher-3dof-reacher-naf-yolo-bottom/09-14_pusher-3dof-reacher-naf-yolo_bottom_2017_09_14_17_52_45_0019/params.pkl'
)


class CombinedNafPolicy(SerializablePolicy, Serializable):
    def __init__(
            self,
            naf_policy_1: NafPolicy,
            naf_policy_2: NafPolicy,
    ):
        Serializable.quick_init(self, locals())
        self.naf_policy_1 = naf_policy_1
        self.naf_policy_2 = naf_policy_2

    # noinspection PyCallingNonCallable
    def get_action(self, obs):
        mu1, P1 = self.naf_policy_1.get_action_and_P_matrix(obs)
        mu2, P2 = self.naf_policy_2.get_action_and_P_matrix(obs)
        inv = np.linalg.inv(P1 + P2)
        return inv * (P1 @ mu1 + P2 @ mu2), {}


def create_policy(variant):
    bottom_snapshot = joblib.load(variant['bottom_path'])
    column_snapshot = joblib.load(variant['column_path'])
    policy = CombinedNafPolicy(
        naf_policy_1=bottom_snapshot['naf_policy'],
        naf_policy_2=column_snapshot['naf_policy'],
    )
    env = bottom_snapshot['env']
    logger.save_itr_params(
        0,
        dict(
            policy=policy,
            env=env,
        )
    )
    path = rollout(
        env,
        policy,
        max_path_length=variant['max_path_length'],
        animated=True,
    )
    env.log_diagnostics([path])
    logger.dump_tabular()


if __name__ == '__main__':
    # exp_prefix = "dev-naf-combine-policies"
    for column in ['left', 'middle', 'right']:
        column = 'right'
        exp_prefix = "combine-naf-policies-bottom-" + column

        variant = dict(
            column=column,
            column_path=column_to_path[column],
            bottom_path=bottom_path,
            max_path_length=300,
        )
        run_experiment(
            create_policy,
            exp_prefix=exp_prefix,
            mode='here',
            variant=variant,
        )
