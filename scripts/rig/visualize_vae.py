from railrl.torch.vae.visualize_conditional_vae import VAEVisualizer, tk, load_dataset

if __name__ == "__main__":
    data_path = \
        "/home/vitchyr/git/railrl/data/doodads3/manual-upload" \
        "/SawyerMultiobjectEnv_N100000_sawyer_init_camera_zoomed_in_imsize84_random_oracle_split_0subsetv2_8cylinders_h100_saveevery5.npy"
    train_data, test_data = load_dataset(data_path)
    model_path = "/home/vitchyr/git/railrl/data/doodads3/01-29-latent-dim-6-2/01-29-latent-dim-6-2_2019_01_29_22_24_54_id000--s93792/params.pkl"
    VAEVisualizer(model_path, train_data, test_data)

    tk.mainloop()
