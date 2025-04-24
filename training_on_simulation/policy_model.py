import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# Set device for computation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)







# Define the ActorCritic network architecture
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        to_pad = (3 - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=to_pad, stride=2),
            nn.GroupNorm(num_channels=12, num_groups=4),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding=to_pad),
            nn.GroupNorm(num_channels=24, num_groups=4),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=to_pad),
            nn.GroupNorm(num_channels=48, num_groups=4),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(4)
        )
        
        
        # Compute the size of the output after the convolution layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim, input_dim)  # Dummy input
            conv_output = self.conv1(dummy_input)
            self.flattened_size = conv_output.view(1, -1).size(1)  # Calculate the flattened size
            
        self.actor = nn.Sequential(
                                  nn.Linear(self.flattened_size, output_dim),
                                  nn.Softmax(dim=-1))
        self.critic = nn.Sequential(
                                    nn.Linear(self.flattened_size, self.flattened_size),
                                    nn.ReLU(),
                                    nn.Linear(self.flattened_size, 1))
        self.terminal = nn.Sequential(nn.Linear(self.flattened_size, 1), nn.Sigmoid())

    def forward(self, state, mask):
        state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(device)
        features = self.conv1(state).view(state.size(0), -1)
        value = self.critic(features)
        probs = self.actor(features)
        dist = CategoricalMasked(logits=torch.log(probs+1e-8), masks = mask)
        terminal_prob = self.terminal(features)
        return dist, value, terminal_prob
