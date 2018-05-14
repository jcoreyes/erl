import matplotlib.pyplot as plt
from railrl.misc import plot_util as plot
plt.style.use("ggplot")
dirs = [
    '/home/vitchyr/git/railrl/data/doodads3/05-12-sawyer-push-and-reach-easy/',
]
f = plot.filter_by_flat_params({
    'replay_buffer_kwargs.fraction_goals_are_env_goals': 0.5,
    'replay_buffer_kwargs.fraction_goals_are_rollout_goals': 0.2,
})
exps = plot.load_exps(dirs, f, suppress_output=False)

plot.split(exps, ["Final  block_distance Mean"],
           [],
           default_vary={"do_state_based_exp": False},
          smooth=plot.ma_filter(10),
          print_final=False, print_min=False, print_plot=True)
plt.title("Pusher2D, Distance to Goal")
plt.show()
# plt.savefig("/home/ashvin/data/s3doodad/media/plots/pusher2d.pdf")
