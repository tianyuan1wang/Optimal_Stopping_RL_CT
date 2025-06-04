import numpy as np
from utils import psnr, angle_range
from reconstructor_rebin_experiment import reconstruct_image_from_sinogram, compute_parallel_sinogram, reconstruct_image_from_sinogram_all, read_parameters



class env():
    
    def __init__(self, reward, path):
        self.n = -1
        self.criteria = 0
        self.n_p = []
        self.reward = reward
        self.path = path
        self.a_init = 0
        self.N_a = 180
        self.angles = angle_range(self.N_a)
        
    def step(self, action, mask):
        # count how many angles
        self.a_start += 1
            
        # transfer the selected action to the radius 
        self.angle_action = self.angles[action]

        # store the selected action
        self.actions_num.append(action)
        
        # store the selected angles    
        self.angles_seq.append(self.angle_action)
        
        # get the current reconstruction
        self.state = reconstruct_image_from_sinogram(self.parallel_sinogram_rebin[self.actions_num,:], self.angles_seq)
        
        # get the termination reward for new state
        self.c_r  = self._get_reward(self.angles_seq, action)
        
        # stop criterion inside the environment: if the maximal number of angles is reached
        if self.a_start > 19:
            self.done = True
            self.mask = mask
         
    
        return np.array(self.state), self.reward, self.c_r, self.done, self.angle_action, self.angles_seq, self.n
           
    
    def reset(self):
        # move to the next sample
        self.n += 1
        # reset how many angles
        self.a_start = self.a_init
        # reset the storage of the selected angles 
        self.angles_seq = []
        # reset the storage of the selected actions 
        self.actions_num = []
        # reset done
        self.done=False 
        # reset the initial state
        self.state = np.zeros((239,239))
        # get parameters from the next sample
        self.proj, self.distance_source_origin, self.distance_origin_detector, self.distance_source_detector, self.detector_pixel_size, self.voxel_size = read_parameters(self.path)
        # rebin the projections from the next sample
        self.parallel_sinogram_rebin = compute_parallel_sinogram(self.proj)
        # reconstruct the image using all rebinned projections as ground truth
        self.P =reconstruct_image_from_sinogram_all(self.parallel_sinogram_rebin)
        
        return self.state
    
    def _get_reward(self,):
        
        # Set reward for each step
        
        self.current_reward = psnr(self.P, self.state)
        
        return self.current_reward
    
    


