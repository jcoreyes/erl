import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")
import pickle
import railrl.torch.pytorch_util as ptu
import torch

from railrl.launchers.experiments.ashvin.rfeatures.rfeatures_model import TimestepPredictionModel
model_class = TimestepPredictionModel
representation_size = 128
output_classes = 20
model_kwargs=dict(
    decoder_distribution='gaussian_identity_variance',
    input_channels=3,
    imsize=224,
    architecture=dict(
        hidden_sizes=[200, 200],
    ),
    delta_features=True,
    pretrained_features=False,
)
model = model_class(
    representation_size,
    # decoder_output_activation=decoder_activation,
    output_classes=output_classes,
    **model_kwargs,
)

model_path = "/home/ashvin/data/s3doodad/facebook/models/rfeatures/multitask1/run2/id2/itr_4000.pt"
# model = load_local_or_remote_file(model_path)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model.to(ptu.device)
model.eval()

traj = np.load("/home/ashvin/code/railrl-private/gitignore/rlbench/demo_door_fixed1/demos5b_10_dict.npy")
goal_image = traj[0]["observations"][-1]["image_observation"]
goal_image = goal_image.reshape(1, 3, 500, 300).transpose([0, 1, 3, 2]) / 255.0
goal_image = goal_image[:, :, 60:, 60:500]
goal_image_pt = ptu.from_numpy(goal_image)
goal_latent = model.encode(goal_image_pt).detach().cpu().numpy().flatten()


s = "/home/ashvin/data/s3doodad/ashvin/rfeatures/rlbench/open-drawer-vision/td3bc-with-state1/run6/id0/"
p = s + "video_0_env.p"
d = pickle.load(open(p, "rb"))

# d = traj

all_zs = [goal_latent]
for i in range(5):
    zs = []
    for j in range(len(d[i]["observations"])):
        img = d[i]["observations"][j]["image_observation"]
        img = img.reshape(-1, 3, 500, 300).transpose([0, 1, 3, 2]) / 255.0
        img = img[:, :, 60:, 60:500]
        img.shape

        pt_img = ptu.from_numpy(img).view(-1, 3, 240, 440)
        z = model.encode(pt_img)
        z = ptu.get_numpy(z)
        zs.append(z)
    all_zs.append(zs)

np.save("/tmp/tmp2.npy", all_zs)


