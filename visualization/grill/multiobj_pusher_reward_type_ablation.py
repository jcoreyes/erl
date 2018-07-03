from visualization.grill.config import (
    output_dir,
    ashvin_base_dir,
    vitchyr_base_dir,
    format_func
)
import matplotlib.pyplot as plt
from railrl.misc import plot_util as plot
from railrl.misc import data_processing as dp

dirs = [ashvin_base_dir + 's3doodad/ashvin/vae/fixed3/sawyer-pusher/vae-dense-multi-logprob/run1',]
logprob = plot.load_exps(dirs, suppress_output=True)
plot.tag_exps(logprob, "name", "logprob")

f = plot.filter_by_flat_params({'replay_kwargs.fraction_goals_are_env_goals': 0.5})
exps = plot.load_exps([ashvin_base_dir + 's3doodad/share/steven/pushing-multipushing/multipusher-reward-variants'], f, suppress_output=True)
plot.tag_exps(exps, "name", "pixel")

dirs = [
    ashvin_base_dir + 's3doodad/ashvin/vae/fixed3/sawyer-pusher/vae-dense-multi3/run1',
       ]
f = plot.filter_by_flat_params({'algo_kwargs.num_updates_per_env_step': 4, "replay_kwargs.fraction_goals_are_env_goals": 0.5})
ours = plot.load_exps(dirs, f, suppress_output=True)
plot.tag_exps(ours, "name", "ours")

plot.comparison(ours + logprob + exps, 
          "Final  total_distance Mean", figsize=(6, 5),
          vary = ["name"],
          default_vary={"reward_params.type": "unknown"},
          smooth=plot.padded_ma_filter(10),
          xlim=(0, 250000), method_order=[1, 0, 2])
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.xlabel("Timesteps")
plt.ylabel("Final Distance to Goal")
plt.title("Visual Pusher, Reward Type Ablation")
plt.legend(["GRiLL", "Log Prob.", "Pixel MSE", ], bbox_to_anchor=(0.49, -0.2), loc="upper center", ncol=4, handlelength=1)
plt.tight_layout()
plt.savefig(output_dir + "pusher_multi_reward_type_ablation.pdf")
print("File saved to", output_dir + "pusher_multi_reward_type_ablation.pdf")
