import torch
from torch.distributions import Categorical, Bernoulli
import pickle
import os
import random
import numpy as np
import torch.optim as optim
import importlib
# Import environment
import env
from policy_model import ActorCritic

# Set seed for reproducibility
seed = 1029
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# Global constants
NUM_EPISODES = 100000
LEARNING_RATE = 0.001
GAMMA = 0.99
note = 'mix_'
IMAGE_SIZE = 239
N_A = 180

# Set device for computation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_results(data, filename, directory='results'):
    """
    Save data to a file within a specified directory.

    Args:
        data: The data to be saved.
        filename: The name of the file to save the data to.
        directory: The directory where the file will be saved.
    """
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Construct the full path to the file
    filepath = os.path.join(directory, filename)

    # Save the data to the file
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {filepath}")


def save_models(model, filename, directory='results'):
    """
    Save data to a file within a specified directory.

    Args:
        data: The data to be saved.
        filename: The name of the file to save the data to.
        directory: The directory where the file will be saved.
    """
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Construct the full path to the file
    filepath = os.path.join(directory, filename)

    torch.save(model.state_dict(), filepath)
    print(f"Data saved to {filepath}")






def run_experiments(num_episodes, N_a, env_e, model, optimizer, gamma=0.99, device='cuda'):
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
            mask[action] = 0 
            # the log prob and entropy
            log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()
            
            # sample the decision: termination or continuation
            terminal = Bernoulli(terminal_prob).sample()

            # move to the next state after selecting action and get reward, done
            next_state, reward, c_r, done, _, _, n = env_e.step(action.item())
            
            # count how many angles
            num_angles += 1
            
            # outputs form the policy model based on the next state: used for calculating the following loss
            _, next_value, next_terminal_prob = model(next_state, mask)
            c_r = torch.tensor([[c_r]], device=device)
            
            # Calculate loss components
            # get the advantage
            advantage = reward + (1 - done) * gamma * next_value * (1 - next_terminal_prob.detach()) + next_terminal_prob.detach() * c_r - value
            # actor loss
            actor_loss = -(log_prob * advantage.detach())
            # critic loss
            critic_loss = advantage.pow(2).mean()
            
            # terminal loss
            last_c_r = torch.tensor([[last_c_r]]).to(device)
            terminal_loss = terminal_prob * (value.detach() - last_c_r) * (1-done)
            
            # combine the loss and update the policy model
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy + terminal_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update for the next round
            score += reward
            state = next_state
            last_c_r = c_r

            # terminate the episode when receive signals from done in environment or terminal policy
            if done or terminal:
                break
        score += c_r.item()
        
     
        # check
        if episode % 100 == 0:
            print("episode", episode, 'score', score)
            print("psnr", c_r, "entropy", entropy.item())
            print('num of angles', num_angles)
        
        # store the policy model every 10000 episodes
        if episode % 10000 == 0:
            save_models(model, f'actor_critic_{episode}')

    save_models(model, f'actor_critic_{episode}')
    




# Main function to orchestrate the experiment
if __name__ == "__main__":
    # experimental costs settings
    rewards = [-0.5]
    
    
    for reward in rewards:
        importlib.reload(env)
        from env import *
        model = ActorCritic(input_dim=IMAGE_SIZE, output_dim=N_A).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        
        env_e = env(reward, IMAGE_SIZE, N_A)
        
        run_experiments(NUM_EPISODES, N_A, env_e, model, optimizer)
   
