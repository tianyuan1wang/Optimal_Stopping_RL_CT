import numpy as np

import astra

from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import griddata, LinearNDInterpolator

from flexdata import data
from flexdata import correct



def rebin_fan2par(sinogram, original_angles, source_obj_distance, new_angles, magnification, interpolation_method='knn'):
    """
    Rebin a fan-beam sinogram into a parallel-beam sinogram with selectable interpolation methods.

    Parameters:
    - sinogram: 2D NumPy array, the fan-beam sinogram.
    - original_angles: 1D array, angles in degrees for the fan-beam sinogram.
    - source_obj_distance: float, distance from source to object.
    - new_angles: 1D array, angles in degrees for the parallel-beam sinogram.
    - magnification: float, magnification factor (source-to-detector ratio).
    - interpolation_method: str, method to use for interpolation ('knn', 'griddata', or 'linearnd').

    Returns:
    - F: 2D NumPy array, the re-binned parallel-beam sinogram.
    """
    # Determine the size of the sinogram
    sinogram = sinogram.T
    n_pixels = sinogram.shape[0]
    n_pixels_new = n_pixels

    # Flip sinogram along the second dimension
    sinogram = np.flip(sinogram, axis=1)

    # Compute detector pixel positions
    pixel_positions = np.arange(-(n_pixels - 1) / 2, (n_pixels - 1) / 2 + 1)

    # Calculate gamma angles
    angles_to_pixels = np.arctan(pixel_positions / (magnification * source_obj_distance))

    # Create meshgrid for beta (angles) and gamma (detector)
    beta, gamma = np.meshgrid(np.radians(original_angles), angles_to_pixels)

    # Calculate t and theta
    t = magnification * source_obj_distance * np.sin(gamma)
    theta = np.degrees(gamma) + np.degrees(beta)

    # Recalculate for new detector pixel positions
    pixel_positions_new = np.arange(-(n_pixels_new - 1) / 2, (n_pixels_new - 1) / 2 + 1)
    angles_to_pixels_new = np.arctan(pixel_positions_new / (magnification * source_obj_distance))

    _, gamma_new = np.meshgrid(np.radians(original_angles), angles_to_pixels_new)
    tnew = magnification * source_obj_distance * np.sin(gamma_new)

    # Define uniform t_parallel grid
    t_para = np.linspace(tnew.min(), tnew.max(), n_pixels_new)
    # Create meshgrid for thetaNewCoord and t_para
    thetaNewCoord, tNewCoord = np.meshgrid(new_angles, t_para)

    # Flatten input data (coordinates and values)
    points = np.column_stack((theta.ravel(), t.ravel()))  # Combine theta and t into coordinate pairs
    values = sinogram.ravel()  # Flatten the sinogram data
    # print('num_points', len(points))
    # print('num_values', len(values))
    
    
    
    # Flatten the target grid for interpolation
    target_points = np.column_stack((thetaNewCoord.ravel(), tNewCoord.ravel()))

    # Interpolation based on the selected method
    if interpolation_method == 'knn':
        # Use KNN for local interpolation
        knn = KNeighborsRegressor(n_neighbors=1, weights='distance')
        knn.fit(points, values)
        F_flat = knn.predict(target_points)
    elif interpolation_method == 'griddata':
        # Use griddata interpolation (e.g., cubic interpolation)
        F_flat = griddata(points, values, target_points, method='cubic', fill_value=0)
    elif interpolation_method == 'linearnd':
        # Use LinearNDInterpolator for scattered data
        interpolant = LinearNDInterpolator(points, values, fill_value=0)
        F_flat = interpolant(target_points)
    else:
        raise ValueError(f"Unsupported interpolation method: {interpolation_method}")

    # Reshape the interpolated result back into the desired grid shape
    F = F_flat.reshape(thetaNewCoord.shape)

    # Flip the result to match the desired orientation
    F = np.flip(F, axis=(0, 1))
    

    return F




def read_parameters(path, detSubSamp=1, angSubSamp=10,  scale=4):
    dark = data.read_stack(path, 'di00', sample=detSubSamp)
    flat = data.read_stack(path, 'io00', sample=detSubSamp)
    proj = data.read_stack(path, 'scan_', skip=angSubSamp, sample=detSubSamp)
        
    geom = data.parse_flexray_scansettings(path, sample=detSubSamp)
        
    # Apply profile correction
    geom = correct.correct(geom,
                               profile='cwi-flexray-2022-10-28',
                               do_print_changes=True)
    geom = correct.correct_vol_center(geom)
        
    distance_source_origin = geom['src2obj']
    distance_origin_detector = geom['det2obj']
    distance_source_detector = geom['src2obj'] + geom['det2obj']
        
    detector_pixel_size = geom['det_pixel']
        
        
    detector_pixel_size *= detSubSamp * scale    
    voxel_size = detector_pixel_size * distance_source_origin/distance_source_detector
        
        
        
    flat = (flat - dark).mean(1)
    proj = (proj - dark) / flat[:, None, :]
    proj = -np.log(proj).astype('float32')
    proj = np.ascontiguousarray(proj)
    
    return proj*100, distance_source_origin, distance_origin_detector, distance_source_detector, detector_pixel_size, voxel_size
    





def fanbeam_sinogram_generator(proj, source_origin_distance, source_detector_distance, 
                                          detSubSamp, detector_pixel_size, 
                                          detector_cols, detector_rows):
    """
    Perform fan-beam projection and reconstruction using ASTRA Toolbox.
    
    Parameters:
    - P: Ground truth image(s), with size (N, M).
    - angles: Array of angles for the projections (e.g., np.linspace(0, 2*np.pi, 360)).
    - SOD: Source-to-object distance.
    - SDD: Source-to-detector distance.
    - detSubSamp: Detector subsampling factor.
    - detector_pixel_size: Size of the detector pixels.
    - detector_cols: Number of columns in the detector.
    - detector_rows: Number of rows in the detector.
    - scale: Scaling factor for the detector.
    - iterations: Number of iterations for the reconstruction algorithm.
    - percentage: The percentage of Gaussian noise to add (default is 5%).
    
    Returns:
    - reconstruction: The reconstructed image from the sinogram.
    """
    
    angles = np.linspace(0,2*np.pi, 361)
    # Step 1: Scale and voxel calculations
    detector_pixel_size *= detSubSamp    
    voxel_size = detector_pixel_size * source_origin_distance / source_detector_distance
    distance_origin_detector = source_detector_distance - source_origin_distance
    distance_source_origin = source_origin_distance

    # Apply scale factor
    detector_pixel_size *= detSubSamp  
    voxel_size = detector_pixel_size * source_origin_distance / source_detector_distance
    distance_origin_detector = source_detector_distance - source_origin_distance

    # Step 2: Create vectors for fan-beam projection geometry
    vectors = np.zeros((len(angles), 6))
    for i in range(len(angles)):
        vectors[i, 0] = np.sin(angles[i]) * distance_source_origin / voxel_size
        vectors[i, 1] = -np.cos(angles[i]) * distance_source_origin / voxel_size
        vectors[i, 2] = -np.sin(angles[i]) * distance_origin_detector / voxel_size
        vectors[i, 3] = np.cos(angles[i]) * distance_origin_detector / voxel_size
        vectors[i, 4] = np.cos(angles[i]) * detector_pixel_size / voxel_size
        vectors[i, 5] = np.sin(angles[i]) * detector_pixel_size / voxel_size


   
    # Step 3: Create projection and volume geometry
    proj_geom = astra.create_proj_geom('fanflat_vec', detector_cols // detSubSamp, vectors)
    
    # Step 4: Create an ASTRA data ID for the ground truth image
    projections_id = astra.data2d.create('-sino', proj_geom, proj[4,:,:]) 
   

    # Step 7: Get the sinogram data
    sinogram = astra.data2d.get(projections_id)  
    
   

    return sinogram, proj_geom


def compute_parallel_sinogram(
    proj,
    detSubSamp=1,
    angSubSamp=10,
    source_origin_distance=224.96754,
    source_detector_distance=449.99152299999996,
    detector_pixel_size=0.14960000051532416,
    detector_cols=956,
    detector_rows=10,
    total_angles=361
):
    """
    Computes the parallel-beam sinogram from a fan-beam sinogram, including masking and averaging.
    
    Parameters:
        detSubSamp (int): Detector subsampling factor.
        angSubSamp (int): Angle subsampling factor.
        scale (int): Scaling factor for processing.
        source_origin_distance (float): Distance between the X-ray source and the origin.
        source_detector_distance (float): Distance between the X-ray source and the detector.
        detector_pixel_size (float): Size of a single detector pixel.
        detector_cols (int): Number of detector columns.
        detector_rows (int): Number of detector rows.
        total_angles (int): Number of angles for projections.

    Returns:
        np.ndarray: The computed parallel-beam sinogram.
    """
    # Compute derived parameters
    source_origin_distance_pixel = source_origin_distance / detector_pixel_size
    source_detector_distance_pixel = source_detector_distance / detector_pixel_size
    magnification = source_detector_distance / source_origin_distance

    # Generate fan-beam angles
    # angles = np.linspace(0, 2 * np.pi, total_angles, endpoint=False)
    
    # Placeholder for fanbeam_sinogram_generator
    sinogram, proj_geom = fanbeam_sinogram_generator(
        proj, source_origin_distance, source_detector_distance, detSubSamp, detector_pixel_size, detector_cols, detector_rows
    )
    

    # Generate parallel beam angles
    angles_input = np.linspace(0, 360, total_angles, endpoint=False)
    new_angles_input = np.linspace(0, 360, total_angles, endpoint=False)

    # Rebin the fan-beam sinogram to a parallel-beam sinogram
    parallel_sinogram_rebin = rebin_fan2par(
        sinogram, angles_input, source_origin_distance_pixel, new_angles_input, magnification
    )
    
    parallel_sinogram_rebin = parallel_sinogram_rebin.T

    # Split and average non-zero values of the sinogram
    array1 = parallel_sinogram_rebin[180:360, :]
    array2 = np.flip(parallel_sinogram_rebin[0:180, :], axis=1)
    
    non_zero_mask1 = array1 != 0
    non_zero_mask2 = array2 != 0
    
    parallel_sinogram_rebin = np.where(
        non_zero_mask1 & non_zero_mask2,  # Both values are non-zero
        (array1 + array2) / 2,           # Average the values
        np.where(non_zero_mask1, array1, array2)  # Replace with the non-zero value
    )

        
    return parallel_sinogram_rebin


def reconstruct_image_from_sinogram_all(
    parallel_sinogram_rebin,
    image_size=239,
    num_projections=None,
    detector_spacing=1.0,
    num_iterations=150,
    min_constraint=0.0,
    scale = 4
):
    """
    Reconstructs an image from a parallel-beam sinogram using ASTRA's SIRT_CUDA method.

    Parameters:
        parallel_sinogram_rebin (np.ndarray): Parallel-beam sinogram rebin output.
        image_size (int): Size of the square reconstruction image (pixels).
        num_projections (int): Number of projection angles (optional, inferred from sinogram if None).
        detector_spacing (float): Spacing between detector pixels.
        num_iterations (int): Number of SIRT iterations for reconstruction.
        min_constraint (float): Minimum constraint for the reconstructed image values.

    Returns:
        np.ndarray: Reconstructed image.
    """
    # Infer number of projections and detectors if not provided
    # parallel_sinogram_rebin = parallel_sinogram_rebin.T
    parallel_sinogram_rebin = parallel_sinogram_rebin[:, ::scale]
  
    
    
    
    
    if num_projections is None:
        num_projections = parallel_sinogram_rebin.shape[0]
        
    num_detectors = parallel_sinogram_rebin.shape[1]
    
    # Generate parallel beam angles
    new_angles_s = np.linspace(0, np.pi, num_projections, endpoint=False)

    # Create projection geometry
    proj_geom = astra.create_proj_geom('parallel', detector_spacing, num_detectors, new_angles_s)

    # Create volume geometry
    vol_geom = astra.create_vol_geom(image_size, image_size)

    # Create projector
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
# rec_sirt = W.reconstruct('SIRT_CUDA', parallel_f_d, iterations=150, extraOptions={'MinConstraint':0.0,'MaxConstraint':1.0})
    # Create the ASTRA OpTomo object
    W = astra.OpTomo(proj_id)

    # Perform SIRT_CUDA reconstruction
    rec_sirt_rebin = W.reconstruct(
        'SIRT_CUDA',
        parallel_sinogram_rebin,
        iterations=num_iterations,
        extraOptions={'MinConstraint': 0.0,
                      'MaxConstraint': 1.0}
    )

    return rec_sirt_rebin


def reconstruct_image_from_sinogram(
    parallel_sinogram_rebin,
    selected_angles,
    image_size=239,
    detector_spacing=1.0,
    num_iterations=150,
    min_constraint=0.0,
    scale = 4
):
    """
    Reconstructs an image from a parallel-beam sinogram using ASTRA's SIRT_CUDA method.

    Parameters:
        parallel_sinogram_rebin (np.ndarray): Parallel-beam sinogram rebin output.
        image_size (int): Size of the square reconstruction image (pixels).
        num_projections (int): Number of projection angles (optional, inferred from sinogram if None).
        detector_spacing (float): Spacing between detector pixels.
        num_iterations (int): Number of SIRT iterations for reconstruction.
        min_constraint (float): Minimum constraint for the reconstructed image values.

    Returns:
        np.ndarray: Reconstructed image.
    """
    # Infer number of projections and detectors if not provided
    # parallel_sinogram_rebin = parallel_sinogram_rebin.T
    parallel_sinogram_rebin = parallel_sinogram_rebin[:, ::scale]
    
    
    
        
    num_detectors = parallel_sinogram_rebin.shape[1]
  
    # Generate parallel beam angles
    # new_angles_s = np.linspace(0, np.pi, num_projections, endpoint=False)

    # Create projection geometry
    proj_geom = astra.create_proj_geom('parallel', detector_spacing, num_detectors, selected_angles)

    # Create volume geometry
    vol_geom = astra.create_vol_geom(image_size, image_size)

    # Create projector
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
# rec_sirt = W.reconstruct('SIRT_CUDA', parallel_f_d, iterations=150, extraOptions={'MinConstraint':0.0,'MaxConstraint':1.0})
    # Create the ASTRA OpTomo object
    W = astra.OpTomo(proj_id)

    # Perform SIRT_CUDA reconstruction
    rec_sirt_rebin = W.reconstruct(
        'SIRT_CUDA',
        parallel_sinogram_rebin,
        iterations=num_iterations,
        extraOptions={'MinConstraint': 0.0,
                      'MaxConstraint': 1.0}
    )

    return rec_sirt_rebin



