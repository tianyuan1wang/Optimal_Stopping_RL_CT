import astra
import numpy as np

def parallel_reconstruction(P, n_p, proj_angles, proj_size, vol_geom, n_iter_sirt=150, percentage=0.05):
    proj_geom = astra.create_proj_geom('parallel', 1.0, proj_size, proj_angles)
    proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
    # construct the OpTomo object
    
    W = astra.OpTomo(proj_id)
    sinogram = W * P
    sinogram = sinogram.reshape([len(proj_angles), proj_size])
    
    n = np.random.normal(0, sinogram[-1].std(), (1, proj_size)) * percentage
    
    
    if len(n_p) != 0:
        n = np.concatenate((n_p, n), axis=0)
    
    sinogram_n = sinogram + n
    
    rec_sirt = W.reconstruct('SIRT_CUDA', sinogram_n, iterations=n_iter_sirt, extraOptions={'MinConstraint':0.0,'MaxConstraint':1.0})
    

    
    return rec_sirt, n


def parallel_reconstruction_all(P, proj_angles, proj_size, vol_geom, n_iter_sirt=150, percentage=0.05):
    proj_geom = astra.create_proj_geom('parallel', 1.0, proj_size, proj_angles)
    proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
    # construct the OpTomo object
    
    W = astra.OpTomo(proj_id)
    sinogram = W * P
    sinogram = sinogram.reshape([len(proj_angles), proj_size])
    
    n = np.random.normal(0, sinogram.std(), (len(proj_angles), proj_size)) * percentage
    
    sinogram_n = sinogram + n
    
    rec_sirt = W.reconstruct('SIRT_CUDA', sinogram_n, iterations=n_iter_sirt, extraOptions={'MinConstraint':0.0,'MaxConstraint':1.0})
    

    
    return rec_sirt
