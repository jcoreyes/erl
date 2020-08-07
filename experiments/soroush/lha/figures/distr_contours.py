from rlkit.launchers.sets.mask_inference import plot_Gaussian
import numpy as np
import matplotlib.pyplot as plt

def save_fig(fig, axs, filename):
    plt.axis('off')
    axs.get_xaxis().set_visible(False)
    axs.get_yaxis().set_visible(False)
    axs.axhline(y=0, color='k')
    axs.axvline(x=0, color='k')
    axs.spines['left'].set_position('zero')
    axs.spines['right'].set_visible(False)
    axs.spines['bottom'].set_position('zero')
    axs.spines['top'].set_visible(False)
    axs.xaxis.set_ticks_position('bottom')
    axs.yaxis.set_ticks_position('left')
    axs.plot((1), (0), ls="", marker=">", ms=10, color="k",
             transform=axs.get_yaxis_transform(), clip_on=False)
    axs.plot((0), (1), ls="", marker="^", ms=10, color="k",
             transform=axs.get_xaxis_transform(), clip_on=False)
    plt.savefig(
        filename,
        bbox_inches='tight', pad_inches=0
    )

fig, axs = plot_Gaussian(
    mu=np.array([1.5, 0]),
    sigma_inv=np.array([[1, 0], [0, 0.02]]),
    bounds=[-2, 2],
    list_of_dims=[[0, 1]],
    add_title=False,
)
save_fig(
    fig, axs,
    '/home/soroush/research/railrl/experiments/soroush/lha/figures/x=1.pdf',
)


fig, axs = plot_Gaussian(
    mu=np.array([0, 0]),
    sigma_inv=np.array([[1, -0.99], [-0.99, 1]]),
    bounds=[-2, 2],
    list_of_dims=[[0, 1]],
    add_title=False,
)
save_fig(
    fig, axs,
    '/home/soroush/research/railrl/experiments/soroush/lha/figures/x=y.pdf',
)


import seaborn as sns
df = sns.load_dataset('iris')
fig, axs = plt.subplots(1, 1, figsize=(6, 6))
sns.kdeplot(
    (df.sepal_width - np.mean(df.sepal_width)) / np.std(df.sepal_width),
    (df.sepal_length - np.mean(df.sepal_length)) / np.std(df.sepal_length),
    cmap="Blues", shade=True, bw=.20
)
plt.xlim([-1, 1])
plt.ylim([-1, 1])
save_fig(
    fig, axs,
    '/home/soroush/research/railrl/experiments/soroush/lha/figures/complex_distr.pdf',
)
