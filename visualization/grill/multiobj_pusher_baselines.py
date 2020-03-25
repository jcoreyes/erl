import matplotlib
from visualization.grill.config import (
    output_dir,
    ashvin_base_dir,
    format_func,
    our_method_name,
    configure_matplotlib,
)
import matplotlib.pyplot as plt
from railrl.visualization import plot_util as plot

configure_matplotlib(matplotlib)

dirs = [
    ashvin_base_dir + 's3doodad/ashvin/vae/fixed3/sawyer-pusher/vae-dense-multi3/run1',
]
f = plot.filter_by_flat_params({
                                   'algo_kwargs.num_updates_per_env_step': 4,
                                   "replay_kwargs.fraction_goals_are_env_goals": 0.5
                               })
ours = plot.load_exps(dirs, f, suppress_output=True)
plot.tag_exps(ours, "name", "ours")

dirs = [
    ashvin_base_dir + 's3doodad/ashvin/vae/fixed3/sawyer-pusher/state-dense-multi1/run1',
]
f = plot.filter_by_flat_params({
                                   'replay_kwargs.fraction_goals_are_env_goals': 0.5,
                                   'algo_kwargs.reward_scale': 1e-4
                               })
oracle = plot.load_exps(dirs, f, suppress_output=True)
plot.tag_exps(oracle, "name", "oracle")

# dsae = plot.load_exps(
#     [ashvin_base_dir + 's3doodad/share/steven/pushing-multipushing/multipusher-reward-variants-spatial'],
#     plot.filter_by_flat_params({'training_mode': 'test'}),
#     suppress_output=True,
# )
dsae = plot.load_exps(
    [ashvin_base_dir + 's3doodad/new-results/multipush-svae'],
    plot.filter_by_flat_params({'training_mode': 'test'}),
    suppress_output=True,
)
plot.tag_exps(dsae, "name", "dsae")

# her = plot.load_exps(
#     [ashvin_base_dir + 's3doodad/share/steven/multipush-her-images'],
#     plot.filter_by_flat_params({'algo': 'ddpg', }),
#     suppress_output=True,
# )
her = plot.load_exps(
    [ashvin_base_dir + 's3doodad/new-results/multipush-her'],
    plot.filter_by_flat_params({'algo': 'ddpg', }),
    suppress_output=True,
)
plot.tag_exps(her, "name", "her")

lr = plot.load_exps([ashvin_base_dir + "s3doodad/share/trainmode_train_data/multi"], suppress_output=True)
plot.tag_exps(lr, "name", "l&r")

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
label_to_color = {
    'ours': colors[0],
    'oracle': colors[3],
    'her': colors[1],
    'dsae': colors[2],
    'l&r': colors[4],
}
plot.comparison(
    ours + oracle + dsae + lr + her,
    ["Final  total_distance Mean"],
    vary=["name"],
    # default_vary={"replay_strategy": "future"},
    smooth=plot.padded_ma_filter(10),
    # xlim=(0, 250000),
    xlim=(0, 500000),
    ylim=(0.15, 0.4),
    # figsize=(7, 3.5),
    figsize=(7.5, 4),
    method_order=[4, 0, 1, 3, 2],
    # label_to_color=label_to_color,
)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
# leg.get_frame().set_alpha(0.9)
plt.xlabel("Timesteps")
plt.ylabel("") # "Final Distance to Goal")
plt.title("Visual Multi-object Pusher Baselines")
plt.legend(
    [
        our_method_name,
        "DSAE",
        "HER",
        "Oracle",
        "L&R",
    ],
           # bbox_to_anchor=(0.49, -0.2), loc="upper center", ncol=5, handlelength=1)
    bbox_to_anchor=(1.0, 0.5), loc="center left",
)
plt.tight_layout()
plt.savefig(output_dir + "multiobj_pusher_baselines.pdf")
print("File saved to", output_dir + "multiobj_pusher_baselines.pdf")
# plt.savefig(output_dir + "multiobj_pusher_baselines_no_legend.pdf")
# print("File saved to", output_dir + "multiobj_pusher_baselines_no_legend.pdf")
# plt.savefig(output_dir + "multiobj_pusher_baselines_legend_right.pdf")
# print("File saved to", output_dir +
#       "multiobj_pusher_baselines_legend_right.pdf")
