from visualization.grill.config import (
    output_dir,
    ashvin_base_dir,
    vitchyr_base_dir,
    format_func
)
import matplotlib.pyplot as plt
from railrl.misc import plot_util as plot
from railrl.misc import data_processing as dp


dirs = [
          ashvin_base_dir + 's3doodad/ashvin/vae/fixed3/sawyer-pusher/vae-dense-multi3/run1',
       ]
f = plot.filter_by_flat_params({'algo_kwargs.num_updates_per_env_step': 4, 'rdim': 16, 'replay_kwargs.fraction_goals_are_rollout_goals': 0.2})
her = plot.load_exps(dirs, f, suppress_output=True)

dirs = [
          ashvin_base_dir + 's3doodad/ashvin/vae/fixed3/sawyer-pusher/vae-dense-multi3/run1',
       ]
f = plot.filter_by_flat_params({'algo_kwargs.num_updates_per_env_step': 4, 'rdim': 16, 'replay_kwargs.fraction_goals_are_rollout_goals': 1.0, 'replay_kwargs.fraction_goals_are_env_goals': 0.0})
norelabel = plot.load_exps(dirs, f, suppress_output=True)
 
dirs = [
     ashvin_base_dir + 's3doodad/ashvin/vae/fixed3/sawyer-pusher/vae-dense-multi3-fullrelabel/run1',
]
fullrelabel = plot.load_exps(dirs, suppress_output=True)
plot.comparison(her + fullrelabel + norelabel, "Final  total_distance Mean", 
           ["replay_kwargs.fraction_goals_are_rollout_goals", "replay_kwargs.fraction_goals_are_env_goals", ],
#            ["training_mode", "replay_kwargs.fraction_goals_are_env_goals", "replay_kwargs.fraction_goals_are_rollout_goals", "rdim"],
           default_vary={"replay_strategy": "future"},
          smooth=plot.padded_ma_filter(10), figsize=(6, 5),
          xlim=(0, 500000), ylim=(0.15, 0.35),  
                method_order=[1, 2, 0, 3])
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.ylabel("")
plt.xlabel("Timesteps")
plt.title("Visual Multi-object Pusher")
leg = plt.legend(["GRiLL", "None", "HER", "VAE", ], bbox_to_anchor=(0.49, -0.2), loc="upper center", ncol=4, handlelength=1)
leg.get_frame().set_alpha(0.9)
plt.tight_layout()
plt.savefig(output_dir + "pusher_multi_relabeling_ablation_b.pdf")
print("File saved to", output_dir + "pusher_multi_relabeling_ablation_b.pdf")
