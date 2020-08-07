import cv2
from torchvision.utils import save_image

from rlkit.envs.vae_wrappers import load_vae
from PIL import Image
import numpy as np
from rlkit.torch import pytorch_util as ptu
import matplotlib.pyplot as plt
# import ipdb; ipdb.set_trace()

im = cv2.imread('goal_train_final_for_vae.png', cv2.IMREAD_UNCHANGED)
# import ipdb; ipdb.set_trace()
im = (im / 255.0).transpose().flatten()
print(im.shape)
vae = load_vae('/home/murtaza/Documents/rllab/railrl/experiments/murtaza/vae/itr_1000.pkl')
tensor = ptu.np_to_var(im)
reconstruction, mu, logvar = vae(tensor)
reconstruction = reconstruction.data.numpy()
reconstruction = np.reshape(reconstruction, (3, 84, 84)).transpose() * 255
cv2.imwrite('reconstruction.png',reconstruction, [cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 1])
