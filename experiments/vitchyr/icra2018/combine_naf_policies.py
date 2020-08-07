import joblib

from rlkit.launchers.launcher_util import run_experiment
from rlkit.policies.composition import AveragerPolicy, CombinedNafPolicy
from rlkit.samplers.util import rollout
from rlkit.core import logger

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


def create_policy(variant):
    bottom_snapshot = joblib.load(variant['bottom_path'])
    column_snapshot = joblib.load(variant['column_path'])
    policy = variant['combiner_class'](
        policy1=bottom_snapshot['naf_policy'],
        policy2=column_snapshot['naf_policy'],
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
        animated=variant['render'],
    )
    env.log_diagnostics([path])
    logger.dump_tabular()


if __name__ == '__main__':
    # exp_prefix = "dev-naf-combine-policies"
    # exp_prefix = "average-naf-policies-bottom"
    exp_prefix = "1-combine-naf-policies"
    for column in [
        'left',
        'middle',
        'right',
    ]:
        new_exp_prefix = "{}-{}".format(exp_prefix, column)

        variant = dict(
            column=column,
            column_path=column_to_path[column],
            bottom_path=bottom_path,
            max_path_length=300,
            combiner_class=CombinedNafPolicy,
            # combiner_class=AveragerPolicy,
            render=False,
        )
        run_experiment(
            create_policy,
            exp_prefix=new_exp_prefix,
            mode='here',
            variant=variant,
        )
