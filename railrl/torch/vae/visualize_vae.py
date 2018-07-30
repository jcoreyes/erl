import tkinter as tk

from railrl.misc.asset_loader import sync_down

import cv2
import pickle
import numpy as np

from PIL import Image, ImageTk
from railrl.torch import pytorch_util as ptu

def load_vae(vae_file):
    if vae_file[0] == "/":
        local_path = vae_file
    else:
        local_path = sync_down(vae_file)
    vae = pickle.load(open(local_path, "rb"))
    # vae = torch.load(local_path, map_location=lambda storage, loc: storage)
    print("loaded", local_path)
    return vae

class VAEVisualizer(object):
    def __init__(self, path, train_dataset, test_dataset):
        self.path = path
        self.vae = load_vae(path)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = 1

        self.master = tk.Tk()

        self.sliders = []
        for i in range(self.vae.representation_size):
            w = tk.Scale(self.master, from_=-3, to=3, orient=tk.HORIZONTAL, resolution=0.01,)
            x, y = (i % 4), 9 + (i // 4)
            w.grid(row=x, column=y)
            self.sliders.append(w)

        self.new_train_image_button = tk.Button(self.master, text="New Training Image", command=self.new_train_image)
        self.new_train_image_button.grid(row=0, column=8)
        self.new_test_image_button = tk.Button(self.master, text="New Testing Image", command=self.new_test_image)
        self.new_test_image_button.grid(row=1, column=8)
        self.reparametrize_button = tk.Button(self.master, text="Reparametrize", command=self.reparametrize)
        self.reparametrize_button.grid(row=2, column=8)

        self.leftpanel = tk.Canvas(self.master, width=84, height=84)
        self.rightpanel = tk.Canvas(self.master, width=84, height=84)
        self.leftpanel.grid(row=0, column=0, columnspan=4, rowspan=4)
        self.rightpanel.grid(row=0, column=4, columnspan=4, rowspan=4)

        self.last_mean = np.zeros((self.vae.representation_size))

        # import pdb; pdb.set_trace()
        self.new_train_image()

        self.master.after(0, self.update)

    def update(self):
        for i in range(self.vae.representation_size):
            self.mean[i] = self.sliders[i].get()
        self.check_change()

        self.master.update()
        self.master.after(10, self.update)

    def get_photo(self, flat_img):
        img = flat_img.reshape((3, 84, 84)).transpose()
        img = (255 * img).astype(np.uint8)
        im = Image.fromarray(img)
        photo = ImageTk.PhotoImage(image=im)

    def new_test_image(self):
        self.new_image(True)

    def new_train_image(self):
        self.new_image(False)

    def new_image(self, test=True):
        if test:
            ind = np.random.randint(0, len(self.test_dataset), (1,))
            self.sample = self.test_dataset[ind, :]
        else:
            ind = np.random.randint(0, len(self.train_dataset), (1,))
            self.sample = self.train_dataset[ind, :]
        img = self.sample.reshape((3, 84, 84)).transpose()
        img = (255 * img).astype(np.uint8)
        self.im = Image.fromarray(img)
        self.leftphoto = ImageTk.PhotoImage(image=self.im)
        self.leftpanel.create_image(0,0,image=self.leftphoto,anchor=tk.NW)

        batch = ptu.np_to_var(self.sample)
        self.mu, self.logvar = self.vae.encode(batch)
        self.z = self.mu
        self.mean = ptu.get_numpy(self.z).flatten()
        self.recon_batch = self.vae.decode(self.z)
        self.update_sliders()
        self.check_change()

    def reparametrize(self):
        self.z = self.vae.reparameterize(self.mu, self.logvar)
        self.mean = ptu.get_numpy(self.z).flatten()
        self.recon_batch = self.vae.decode(self.z)
        self.update_sliders()
        self.check_change()

    def check_change(self):
        if not np.allclose(self.mean, self.last_mean):
            z = ptu.np_to_var(self.mean[:, None].transpose())
            self.recon_batch = self.vae.decode(z)
            self.update_reconstruction()
            self.last_mean = self.mean.copy()

    def update_sliders(self):
        for i in range(self.vae.representation_size):
            self.sliders[i].set(self.mean[i])

    def update_reconstruction(self):
        recon_numpy = ptu.get_numpy(self.recon_batch)
        img = recon_numpy.reshape((3, 84, 84)).transpose()
        img = (255 * img).astype(np.uint8)
        self.rightim = Image.fromarray(img)
        self.rightphoto = ImageTk.PhotoImage(image=self.rightim)
        self.rightpanel.create_image(0,0,image=self.rightphoto,anchor=tk.NW)

    def get_batch(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        ind = np.random.randint(0, len(dataset), self.batch_size)
        samples = dataset[ind, :]
        # if self.normalize:
        #     samples = ((samples - self.train_data_mean) + 1) / 2
        return ptu.np_to_var(samples)

def load_dataset(filename, test_p=0.9):
    dataset = np.load(filename)
    N = len(dataset)
    n = int(N * test_p)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset

if __name__ == "__main__":
    # from railrl.torch.vae.sawyer2d_push_new_easy_data_wider import generate_vae_dataset
    # train_data, test_data, info = generate_vae_dataset(
    #     N=10000
    # )
    data_path = "/tmp/SawyerPickAndPlaceEnvYZ_6000_sawyer_pick_and_place_camera_oracleTruenext_dataset.npy"
    train_data, test_data = load_dataset(data_path)
    model_path = "/home/steven/vae.pkl"
    VAEVisualizer(model_path, train_data, test_data)

    tk.mainloop()
