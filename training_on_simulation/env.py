import numpy as np

from matplotlib import pyplot as plt

import astra

from synthetic_data_generation import generate_and_save_data
from utils import psnr, angle_range
from reconstructor import parallel_reconstruction


P_t, L_t, P_f, L_f, P_fo, L_fo = generate_and_save_data('simulation_model')
P_all = P_fo + P_t + P_f



a_start = 0
image_size = 239
N_a = 180
angles = angle_range(N_a)
image_size = 239
proj_size = image_size
vol_geom = astra.create_vol_geom(image_size, image_size)



class env():
    
    def __init__(self, reward):
        # get a sample from the dataset randomly
        self.n = np.random.randint(0,6000)
        # set the experimental cost (negative value)
        self.reward = reward
 
        
    def step(self, action):
        # count how many angles
        self.a_start += 1
        
        # transfer the selected action to the radius 
        self.angle_action = angles[action]
        
        # store the selected action
        self.actions_num.append(action)

        # store the selected angles    
        self.angles_seq.append(self.angle_action)

        # get the current reconstruction and updated the noisy mearsurements
        self.state, self.n_p = parallel_reconstruction(P_all[self.n], self.n_p, self.angles_seq, proj_size, vol_geom)

        # get the termination reward for new state
        self.c_r  = self._get_reward()
      
        # stop criterion inside the environment: if the maximal number of angles is reached
        if self.a_start > 19:
            self.done = True
         
        return np.array(self.state), self.reward, self.c_r, self.done, self.angle_action, self.angles_seq, self.n
           
    
    def reset(self):
        # get a sample from the dataset randomly
        self.n = np.random.randint(0,6000)
        # reset how many angles
        self.a_start = 0
        # reset the storage of the selected angles    
        self.angles_seq = []
        # reset the storage of the selected actions 
        self.actions_num = []
        # reset done
        self.done=False 
        # reset the storage of the measurements
        self.n_p = []
        # reset the initial state
        self.state = np.zeros((image_size,image_size))
        
        return self.state
    
    def _get_reward(self,):
        
        # Set the terminal reward for each step
        self.current_reward = psnr(P_all[self.n], self.state)
        
      
        
        return self.current_reward
    
