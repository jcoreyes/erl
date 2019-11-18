from railrl.demos.collect_demo import collect_demos
from railrl.misc.asset_loader import load_local_or_remote_file

if __name__ == '__main__':
    data = load_local_or_remote_file('/home/murtaza/research/railrl/data/doodads3/11-16-pickup-state-td3-sweep-params-policy-update-period/11-16-pickup_state_td3_sweep_params_policy_update_period_2019_11_17_00_26_46_id000--s79057/params.pkl')
    env = data['evaluation/env']
    policy = data['trainer/trained_policy']
    collect_demos(env, policy, "data/local/demos/pickup_demos_1000.npy", N=1000, horizon=50, threshold=.02, add_action_noise=False, key='obj_distance')
