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
    ashvin_base_dir + 's3doodad/ashvin/vae/fixed3/sawyer-pusher/vae-dense-multi-logprob/run1', ]
logprob = plot.load_exps(dirs, suppress_output=True)
plot.tag_exps(logprob, "name", "logprob")

f = plot.filter_by_flat_params(
    {'replay_kwargs.fraction_goals_are_env_goals': 0.5})
exps = plot.load_exps([
    ashvin_base_dir + 's3doodad/share/steven/pushing-multipushing/multipusher-reward-variants'],
    f, suppress_output=True)
plot.tag_exps(exps, "name", "pixel")

dirs = [
    ashvin_base_dir + 's3doodad/ashvin/vae/fixed3/sawyer-pusher/vae-dense-multi3/run1',
]
f = plot.filter_by_flat_params({
                                   'algo_kwargs.num_updates_per_env_step': 4,
                                   "replay_kwargs.fraction_goals_are_env_goals": 0.5
                               })
ours = plot.load_exps(dirs, f, suppress_output=True)
plot.tag_exps(ours, "name", "ours")

plot.comparison(
    ours + logprob + exps,
    "Final  total_distance Mean",
    figsize=(7.5, 4),
    vary=["name"],
    default_vary={"reward_params.type": "unknown"},
    smooth=plot.padded_ma_filter(10),
    xlim=(0, 250000),
    method_order=[1, 0, 2],
)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.xlabel("Timesteps")
plt.ylabel("")
plt.title("Visual Multi-object Pusher")
plt.legend([our_method_name, "Log Prob.", "Pixel MSE", ],
    bbox_to_anchor=(1.0, 0.5), loc="center left",)
plt.tight_layout()
plt.savefig(output_dir + "multiobj_pusher_reward_type_ablation.pdf")
print("File saved to", output_dir + "multiobj_pusher_reward_type_ablation.pdf")
