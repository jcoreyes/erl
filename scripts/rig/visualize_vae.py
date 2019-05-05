from railrl.torch.vae.visualize_vae import (
    VAEVisualizer, ConditionalVAEVisualizer, tk, load_dataset
)

from railrl.torch import pytorch_util as ptu
# ptu.set_gpu_mode(True)

def visualize(args):
    data_path = args.data_path
    model_path = args.model_path
    train_data, test_data = load_dataset(data_path)

    visualizer = ConditionalVAEVisualizer if conditional else VAEVisualizer
    visualizer(model_path, train_data, test_data)

    tk.mainloop()

if __name__ == "__main__":
    # data_path = "/tmp/SawyerDoorHookResetFreeEnv-v1_N2_sawyer_door_env_camera_v0_imsize48_random_oracle_split_0.npy"
    # model_path = "/home/ashvin/data/s3doodad/ashvin/arl/pusher/skewfit/run0/id0/vae.pkl"

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path to the .npy data file')
    parser.add_argument('model_path', type=str, help='path to the .pkl model file')
    parser.add_argument('--conditional', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
