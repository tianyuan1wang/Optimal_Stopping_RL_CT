import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli
import pickle
import os
import random
import numpy as np
import importlib
from scipy import ndimage
import pickle

from policy_model import ActorCritic

# Global constants
NUM_EPISODES = 1

# Set device for computation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Import environment
from env_test import env



def run_experiments(num_episodes, N_a, env_e, model, device='cuda'):
    for episode in range(num_episodes):
        # reset environment
        state = env_e.reset()
        # mask for selected angles
        mask = torch.ones(N_a, dtype=torch.bool, device=device)
        # reset at the begining of an episode
        score, num_angles, last_c_r, terminal = 0, 0, 0, False
        while True:
            # outputs from the policy model
            dist, value, terminal_prob = model(state, mask)
            # sample an action from the distribution
            action = dist.sample()
            # update the mask for selected angles
            mask[action] = 0  # Update mask
            # the entropy
            entropy = dist.entropy().mean()
            # sample the decision: termination or continuation
            terminal = Bernoulli(terminal_prob).sample()

            # move to the next state after selecting action and get reward, done
            next_state, reward, c_r, done, _, _, n = env_e.step(action.item(), mask)
            # count how many angles
            num_angles += 1

            c_r = torch.tensor([[c_r]], device=device)
            
            last_c_r = torch.tensor([[last_c_r]]).to(device)
            
            # update for the next round
            score += reward

            state = next_state
            last_c_r = c_r

            if done or terminal:
                break
        score += c_r.item()
        

        if episode % 100 == 0:
            print("episode", episode, 'score', score)
            print("psnr", c_r, "entropy", entropy.item())
            print('num of angles', num_angles, 'n', env_e.n)
            




# Main function to orchestrate the experiment
def process_paths(paths, numbers, rewards, filename, noise, prefix, save_dir='plot_data_N5_test'):
    """
    Process a list of paths with given parameters to run experiments and save results.

    Args:
        paths (list): List of paths to process.
        numbers (list): List of number identifiers for the runs.
        rewards (list): List of rewards.
        filename (str): File name for the model.
        noise (str): Noise level identifier.
        scale (str): Scaling identifier.
        prefix (str): Prefix for naming output files (e.g., 'T', 'TN', 'P', 'PN').
        note (str): Additional note for labeling.
    """
    num_a_all_list = []
    psnr_all_list = []
    angles_list = []

    for number in numbers:
        psnr_all = []
        num_a_all = []
        angles_all =[]

        for k, path in enumerate(paths, start=1):
            # Reload and import environment
            importlib.reload(env)
            from env_test import env
            
            # get the path to load the trained model
            addition_name = f'0{number}'
            folder = f'R0.{number}_soft'
            file_path = os.path.join(folder, filename + addition_name)

            # load the trained model
            if os.path.exists(file_path):
                model = ActorCritic(input_dim=239, output_dim=180).to(device)
                model.load_state_dict(torch.load(file_path, map_location=device))
                model.eval()
                print(f"Data from {file_path}")
            else:
                print(f"No file found at {file_path}")
                continue

            # Create environment and run experiments
            env_e = env(rewards[0], path)
            run_experiments(
                NUM_EPISODES, 180, env_e, model
            )

        # Print results
        print(
            f"{prefix} | num_a_all: {np.array(num_a_all)}, "
            f"sum: {np.sum(num_a_all)}, mean: {np.mean(num_a_all)}, std: {np.std(num_a_all)}"
        )

        # Save results to the specified directory
        num_a_path = os.path.join(save_dir, f"NUM_A_{prefix}_{number}_{noise}.pkl")
        psnr_path = os.path.join(save_dir, f"PSNR_{prefix}_{number}_{noise}.pkl")
        angles_path = os.path.join(save_dir, f"Angles_{prefix}_{number}_{noise}.pkl")

        with open(num_a_path, "wb") as f:
            pickle.dump(num_a_all, f)
        with open(psnr_path, "wb") as f:
            pickle.dump(psnr_all, f)

        with open(angles_path, "wb") as f:
            pickle.dump(angles_all, f)

        num_a_all_list.append(np.array(num_a_all))
        psnr_all_list.append(np.array(psnr_all))

    return num_a_all_list, psnr_all_list


if __name__ == "__main__":
    note = "mix_"
    ep = 80000

    # Directories
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "AngleSelection3"))
    noise_1 = "600"
    noise_2 = "100"

    # Paths
    k_s, k_e = 1, 13
    pathsT = [f"{base_dir}/triangle{i}_{noise_1}/" for i in range(k_s, k_e)]
    pathsTN = [f"{base_dir}/triangle{i}_{noise_2}/" for i in range(k_s, k_e)]
    pathsP = [f"{base_dir}/pentagon{i}_{noise_1}/" for i in range(k_s, k_e)]
    pathsPN = [f"{base_dir}/pentagon{i}_{noise_2}/" for i in range(k_s, k_e)]

    # Parameters
    numbers = [8,9]
    rewards = [-0.8,-0.9]
    filename = f"actor_critic_{ep}_1029_mix_"

    # Process each group
    NUM_A_TA, PSNR_TA = process_paths(pathsT, numbers, rewards, filename, noise_1, f"T{ep}")
    NUM_A_NTA, PSNR_NTA = process_paths(pathsTN, numbers, rewards, filename, noise_2, f"T{ep}")
    NUM_A_PA, PSNR_PA = process_paths(pathsP, numbers, rewards, filename, noise_1, f"P{ep}")
    NUM_A_NPA, PSNR_NPA = process_paths(pathsPN, numbers, rewards, filename, noise_2, f"P{ep}")


    print("NUM_A_TA:", [np.mean(arr) for arr in NUM_A_TA], "std", [np.std(arr) for arr in NUM_A_TA])
    print("NUM_A_PA:", [np.mean(arr) for arr in NUM_A_PA], "std", [np.std(arr) for arr in NUM_A_PA])
    print("NUM_A_NTA:", [np.mean(arr) for arr in NUM_A_NTA], "std", [np.std(arr) for arr in NUM_A_NTA])
    print("NUM_A_NPA:", [np.mean(arr) for arr in NUM_A_NPA], "std", [np.std(arr) for arr in NUM_A_NPA])
    
     
