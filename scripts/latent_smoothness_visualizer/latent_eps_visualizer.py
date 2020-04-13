import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import tkinter as tk
from railrl.misc.asset_loader import sync_down
from railrl.misc.asset_loader import load_local_or_remote_file
import torch
import pickle
import numpy as np
import io
import skvideo.io
from PIL import Image, ImageTk
from railrl.torch import pytorch_util as ptu
from railrl.data_management.dataset  import \
        TrajectoryDataset, ImageObservationDataset, InitialObservationDataset
from railrl.data_management.images import normalize_image, unnormalize_image

def load_model(model_file):
    if model_file[0] == "/":
        local_path = model_file
    else:
        local_path = sync_down(model_file)
    # vae = pickle.load(open(local_path, "rb"))
    model = torch.load(local_path, map_location='cpu')
    print("loaded", local_path)
    model.to("cpu")
    return model











class LatentVisualizer(object):
    def __init__(self, path, env):
        self.vae = load_vae(path)
        self.env = env




    def get_states(self):
    	return np.array([-0.2, -0.2, 0.15, 0.15], [-0.2, -0.2, 0.13, 0.15], [-0.2, -0.2,0.15, 0.13])

    def get_eps_ball(self, state, eps, num=5):


    def get_images(self, states):
    	self.env.reset()

    	for i in range(states.shape[0]):
    		gaol = dict(state_desired_goal=states[i])
    		self.env.set_to_goal(goal)
    		obs self.env._get_obs()


    def sample_latents(self):
    	self.env._sample_position()

    def load_dataset(filename, test_p=0.9):
        dataset = np.load(filename)
        N = len(dataset)
        n = int(N * test_p)
        train_dataset = dataset[:n, :]
        test_dataset = dataset[n:, :]
        return train_dataset, test_dataset

    def update(self):
        for i in range(self.vae.representation_size):
            self.mean[i] = self.sliders[i].get()
        self.check_change()

        self.master.update()
        self.master.after(10, self.update)

    def get_photo(self, flat_img):
        img = flat_img.reshape((3, 48, 48)).transpose()
        img = (255 * img).astype(np.uint8)
        im = Image.fromarray(img)
        photo = ImageTk.PhotoImage(image=im)

    def new_test_image(self):
        self.new_image(True)

    def new_train_image(self):
        self.new_image(False)

    def new_image(self, test=True):
        if test:
            self.batch = self.test_dataset.random_batch(1)
        else:
            self.batch = self.train_dataset.random_batch(1)
        self.sample = self.batch["x_t"]
        #self.sample = self.train_dataset[ind, :] / 255
        img = unnormalize_image(ptu.get_numpy(self.sample).reshape((3, 48, 48)).transpose())
        #img = self.sample.reshape((3, 48, 48)).transpose()
        #img = (255 * img).astype(np.uint8)
        # img = img.astype(np.uint8)
        self.im = Image.fromarray(img)
        #self.leftphoto = ImageTk.PhotoImage(image=self.im)
        #self.leftpanel.create_image(0,0,image=self.leftphoto,anchor=tk.NW)

        self.mu, self.logvar = self.vae.encode(self.sample)
        self.z = self.mu
        self.mean = ptu.get_numpy(self.z).flatten()
        self.home_mean = self.mean.copy()
        self.recon_batch = self.vae.decode(self.z)[0]
        self.update_sliders()
        self.check_change()

    def reparametrize(self):
        self.z = self.vae.reparameterize((self.mu, self.logvar))
        self.mean = ptu.get_numpy(self.z).flatten()
        self.recon_batch = self.vae.decode(self.z)[0]
        self.update_sliders()
        self.check_change()

    def check_change(self):
        if not np.allclose(self.mean, self.last_mean):
            z = ptu.from_numpy(self.mean[:, None].transpose())
            self.recon_batch = self.vae.decode(z)[0]
            self.update_reconstruction()
            self.last_mean = self.mean.copy()

    def update_sliders(self):
        for i in range(self.vae.representation_size):
            self.sliders[i].set(self.mean[i])

    def update_reconstruction(self):
        recon_numpy = ptu.get_numpy(self.recon_batch)
        img = recon_numpy.reshape((3, 48, 48)).transpose()
        img = (255 * img).astype(np.uint8)
        # img = img.astype(np.uint8)
        self.rightim = Image.fromarray(img)
        self.rightphoto = ImageTk.PhotoImage(image=self.rightim)
        self.rightpanel.create_image(0,0,image=self.rightphoto,anchor=tk.NW)

    def get_batch(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        ind = np.random.randint(0, len(dataset), self.batch_size)
        samples = dataset[ind, :]
        return ptu.from_numpy(samples)


    def sweep_element(self):
        data = [np.copy(self.mean)]
        # self.rightim.save('/home/ashvin/ros_ws/src/railrl-private/visualizer/vae/img_0.jpg')
        for i in range(40):
            for k in self.sweep_i:
                if np.random.uniform() < 0.5:
                    sign = 1
                else:
                    sign = -1
                if self.mean[k] >= 3:
                    sign = -1
                if self.mean[k] < -3:
                    sign = 1
                self.mean[k] += sign * 0.25
                self.sliders[k].set(self.mean[k])
            self.check_change()
            self.rightim.save('/home/ashvin/ros_ws/src/railrl-private/visualizer/vae/img_{}.jpg'.format(i + 1))
            data.append(np.copy(self.mean))
            self.master.after(100, self.sweep_element)
        # np.save('/home/ashvin/ros_ws/src/railrl-private/visualizer/vae/latents.npy', np.array(data))
        self.mean = self.original_mean


    def sweep(self):
        self.original_mean = self.mean.copy()
        self.sweep_i = [i for i in range(self.vae.representation_size)] #temp
        self.sweep_k = 0
        self.master.after(100, self.sweep_element)


    def set_home(self):
        self.home_mean = self.mean.copy()

    def sweep_home(self):
        # decode as we interplote from self.home_mean -> self.mean
        frames = []
        for i, t in enumerate(np.linspace(0, 1, 25)):
            z = t * self.home_mean + (1 - t) * self.mean
            print(t, z)
            z = ptu.from_numpy(z[:, None].transpose())
            recon_batch = self.vae.decode(z)[0]
            recon_numpy = ptu.get_numpy(recon_batch)
            img = recon_numpy.reshape((3, 48, 48)).transpose()
            img = (255 * img).astype(np.uint8)
            frames.append(img)
            
        frames += frames[::-1]
        skvideo.io.vwrite("tmp/vae/dog/%d.mp4" % self.saved_videos, frames)
        self.saved_videos += 1








if __name__ == "__main__":
    # from railrl.torch.vae.sawyer2d_push_new_easy_data_wider import generate_vae_dataset
    # train_data, test_data, info = generate_vae_dataset(
    #     N=10000
    # )
    data_path = "/home/ashvin/Desktop/sim_puck_data.npy"
    train_data, test_data = load_dataset(data_path)
    #model_path = "/home/ashvin/data/rail-khazatsky/sasha/cond-rig/hyp-tuning/tuning/run550/id1/vae.pkl"
    model_path = "/home/ashvin/data/sasha/cond-rig/hyp-tuning/dropout/run1/id0/itr_100.pkl"
    ConditionalVAEVisualizer(model_path, train_data, test_data)

    tk.mainloop()