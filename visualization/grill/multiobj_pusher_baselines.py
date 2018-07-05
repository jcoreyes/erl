from visualization.grill.config import (
    output_dir,
    ashvin_base_dir,
    vitchyr_base_dir,
    format_func,
)
import matplotlib.pyplot as plt
from railrl.misc import plot_util as plot

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
f = plot.filter_by_flat_params({'training_mode': 'test'})
dsae = plot.load_exps([
                          ashvin_base_dir + 's3doodad/share/steven/pushing-multipushing/multipusher-reward-variants-spatial'],
                      f, suppress_output=True)
plot.tag_exps(dsae, "name", "dsae")
# lr = plot.load_exps(["/home/ashvin/data/s3doodad/share/trainmode_train_data/multi"], suppress_output=True)
# plot.tag_exps(lr, "name", "l&r")
f = plot.filter_by_flat_params({'algo': 'ddpg', })
her = plot.load_exps(
    [ashvin_base_dir + 's3doodad/share/steven/multipush-her-images'], f,
    suppress_output=True)
plot.tag_exps(her, "name", "her")
her2 = plot.load_exps(
    [ashvin_base_dir + 's3doodad/share/steven/multipush-her-images'], f,
    suppress_output=True)
plot.tag_exps(her2, "name", "l&r")

plot.comparison(
    ours + oracle + dsae + her + her2,
    ["Final  total_distance Mean"],
    vary=["name"],
    default_vary={"replay_strategy": "future"},
    smooth=plot.padded_ma_filter(10),
    xlim=(0, 250000),
    ylim=(0.15, 0.3),
    figsize=(7, 3.5),
    # figsize=(6, 5),
    method_order=[4, 0, 1, 3, 2],
)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
# leg.get_frame().set_alpha(0.9)
plt.xlabel("Timesteps")
plt.ylabel("Final Distance to Goal")
plt.title("Visual Multi-object Pusher Baselines")
plt.legend(["GRiLL", "DSAE", "HER", "Oracle", "L&R", ],
           # bbox_to_anchor=(0.49, -0.2), loc="upper center", ncol=5, handlelength=1)
           bbox_to_anchor=(1.0, 0.5), loc="center left")
# plt.legend([])
plt.tight_layout()
# plt.savefig(output_dir + "multiobj_pusher_baselines.pdf")
# print("File saved to", output_dir + "multiobj_pusher_baselines.pdf")
# plt.savefig(output_dir + "multiobj_pusher_baselines_no_legend.pdf")
# print("File saved to", output_dir + "multiobj_pusher_baselines_no_legend.pdf")
plt.savefig(output_dir + "multiobj_pusher_baselines_legend_right.pdf")
print("File saved to", output_dir +
      "multiobj_pusher_baselines_legend_right.pdf")
