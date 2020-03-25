import matplotlib
from visualization.grill.config import (
    output_dir,
    ashvin_base_dir,
    vitchyr_base_dir,
    format_func,
    configure_matplotlib,
)
import matplotlib.pyplot as plt
from railrl.visualization import plot_util as plot

configure_matplotlib(matplotlib)

dirs = [
    ashvin_base_dir + 's3doodad/ashvin/vae/fixed3/sawyer-pusher/state-dense-wider2/run1',
]
f = plot.filter_by_flat_params({'algo_kwargs.num_updates_per_env_step': 4, 'replay_kwargs.fraction_goals_are_env_goals': 0.0})
oracle = plot.load_exps(dirs, f, suppress_output=True)
plot.tag_exps(oracle, "name", "oracle")

dirs = [
    ashvin_base_dir + 's3doodad/ashvin/vae/fixed3/sawyer-pusher/vae-dense-wider3/run1',
    ashvin_base_dir + 's3doodad/ashvin/vae/fixed3/sawyer-pusher/vae-dense-wider3-relabeling/run1',
       ]
f = plot.filter_by_flat_params({'algo_kwargs.num_updates_per_env_step': 4, "replay_kwargs.fraction_goals_are_env_goals": 0.5, "replay_kwargs.fraction_goals_are_rollout_goals": 0.2})
ours = plot.load_exps(dirs, f, suppress_output=True)
plot.tag_exps(ours, "name", "ours")

# I think this is the right data for HER but we should check with Steven... this line looks different than the original plot
f = plot.filter_by_flat_params({
    'replay_kwargs.fraction_goals_are_env_goals': 0.0,
    'vae_wrapped_env_kwargs.reward_params.type': 'latent_distance',
})
# her = plot.load_exps(
#     [ashvin_base_dir + 's3doodad/share/steven/pushing-multipushing/pusher-reward-variants'],
#     f,
#     suppress_output=True,
# )
her = plot.load_exps(
    [ashvin_base_dir + 's3doodad/new-results/push-her'],
    # f,
    suppress_output=True,
)
plot.tag_exps(her, "name", "her")

# The below need to be updated with the right data....
f = plot.filter_by_flat_params({
    'replay_kwargs.fraction_goals_are_rollout_goals': 1.0,
    'training_mode': 'train',
    'algo_kwargs.reward_scale': 0.01,
    'rdim': 4
})
# dsae = plot.load_exps(
#     [ashvin_base_dir + 's3doodad/share/steven/pushing-multipushing/pusher-reward-variants-spatial/run1'],
#     f,
#     suppress_output=True,
# )
dsae = plot.load_exps(
    [ashvin_base_dir + 's3doodad/new-results/pusher-spatial'],
    f,
    suppress_output=True,
)
plot.tag_exps(dsae, "name", "dsae")

#lr = plot.load_exps(["/home/vitchyr/git/railrl/data/papers/nips2018/autoencoder_result/05-26-sawyer-single-push-autoencoder-ablation-final/"], suppress_output=True)


lr = plot.load_exps(
    [vitchyr_base_dir + "papers/nips2018/06-06-lr-baseline-pusher-0.2-range/"],
    plot.filter_by_flat_params({
        'rdim': '16--lr-1e-3',
    }),
    suppress_output=True,
)
plot.tag_exps(lr, "name", "l&r")

plot.comparison(
    #lr + ours + oracle + her + dsae,
    ours + oracle + her + lr + dsae,
    ["Final  puck_distance Mean", "Final  hand_distance Mean"],
    vary=["name"],
    smooth=plot.padded_ma_filter(10),
    #method_order=[4, 0, 1, 3, 2],
    ylim=(0.1, 0.28),
    # xlim=(0, 250000),
    xlim=(0, 500000),
    figsize=(6, 4),
)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.xlabel("Timesteps")
plt.ylabel("") # "Final Distance to Goal")
plt.title("Visual Pusher Baselines")
plt.legend([]) # [our_method_name, "DSAE", "HER", "Oracle", "L&R", ])

plt.tight_layout()
plt.savefig(output_dir + "pusher_baselines.pdf")
print("File saved to", output_dir + "pusher_baselines.pdf")

