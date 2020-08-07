import torch
import numpy as np
import pickle
from rlkit.misc.asset_loader import load_local_or_remote_file
import rlkit.torch.pytorch_util as ptu

from torchvision.utils import save_image
from rlkit.data_management.images import normalize_image

# dataset_path = "/tmp/SawyerMultiobjectEnv_N100000_sawyer_pusher_camera_upright_v2_imsize48_random_oracle_split_0.npy"
# cvae_path = "/home/ashvin/data/s3doodad/ashvin/corl2019/pusher/color2/run102/id0/itr_500.pkl"
# vae_path = "/home/ashvin/data/s3doodad/ashvin/corl2019/pusher/vae1/run102/id0/itr_500.pkl"

# dataset_path = "/tmp/Multiobj2DEnv_N100000_sawyer_pusher_camera_upright_v2_imsize48_random_oracle_split_0.npy"
# cvae_path = "/home/ashvin/data/s3doodad/ashvin/corl2019/pcvae/pointmass/dcvae1/run102/id0/itr_200.pkl"
# vae_path = "/home/ashvin/data/s3doodad/ashvin/corl2019/pcvae/pointmass/vae1/run102/id0/itr_100.pkl"
# prefix = "pointmass1_"

dataset_path = "/tmp/Point2DWallEnv_N10000_sawyer_door_env_camera_v0_imsize48_random_oracle_split_0pointmass_dcvae_3.npy"
cvae_path = "/home/ashvin/data/s3doodad/datasets/pointmass/vae_pointmass_wall3.pkl"
vae_path = "/home/ashvin/data/s3doodad/ashvin/corl2019/offpolicy/pointmass/vae3/run0/id0/itr_1000.pkl"
prefix = "wall_pointmass1_"
N_ROWS = 3



dataset = np.load(dataset_path)
dataset = dataset.item()

imlength = 6912
imsize = 48

N = len(dataset['observations'])
t = int(0.9 * N)

x_0 = dataset['observations'][t:t+8, 0, :] / 255.0 # test set
x_t = dataset['observations'][t:t+8, 1, :] / 255.0 # test set

# import ipdb; ipdb.set_trace()

n = x_0.shape[0]

x0 = ptu.from_numpy(x_0)

# cvae = cvae.cpu()
# latent_goal = cvae.encode(ptu.from_numpy(x.reshape(1,-1)), x0, distrib=False)
# decoded_goal, _ = cvae.decode(latent_goal,x0)

all_imgs = [x0.narrow(start=0, length=imlength, dim=1).contiguous().view(-1, 3, imsize, imsize).transpose(2, 3), ]
comparison = torch.cat(all_imgs)
save_dir = "/home/ashvin/data/s3doodad/share/multiobj/%sx0.png" % prefix
save_image(comparison.data.cpu(), save_dir, nrow=8)





vae = load_local_or_remote_file(vae_path).to("cpu")
vae.eval()

model = vae
all_imgs = []
for i in range(N_ROWS):
    latent = ptu.randn(n, model.representation_size) # model.sample_prior(self.batch_size, env)
    samples = model.decode(latent)[0]
    all_imgs.extend([
        samples.view(
            n,
            3,
            imsize,
            imsize,
        )[:n].transpose(2, 3)])
comparison = torch.cat(all_imgs)
save_dir = "/home/ashvin/data/s3doodad/share/multiobj/%svae_samples.png" % prefix
save_image(comparison.data.cpu(), save_dir, nrow=8)





cvae = load_local_or_remote_file(cvae_path).to("cpu")
cvae.eval()

model = cvae
all_imgs = []
for i in range(N_ROWS):
    latent = model.sample_prior(n, x0)
    samples = model.decode(latent)[0]
    all_imgs.extend([
        samples.view(
            n,
            3,
            imsize,
            imsize,
        )[:n].transpose(2, 3)])
comparison = torch.cat(all_imgs)
save_dir = "/home/ashvin/data/s3doodad/share/multiobj/%scvae_samples.png" % prefix
save_image(comparison.data.cpu(), save_dir, nrow=8)
