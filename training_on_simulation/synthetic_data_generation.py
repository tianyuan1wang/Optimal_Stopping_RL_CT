import numpy as np
from skimage import draw
import os
import pickle  # For saving multiple arrays together
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import shift, rotate
import torch
import random

# === Configuration ===
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()



def get_config(size=239, num_samples=2000):
    image_scale = size / 512
    return {
        'size': size,
        'num_samples': num_samples,
        'image_scale': image_scale,
        'rotation_range': np.linspace(0, 180, 36, endpoint=False)
    }

# === Shape Functions (same as before) ===
def create_parallelogram(center_x, center_y, radius, angle, size):
    """
    Create a parallelogram by joining the shortest edges of two isosceles right triangles,
    ensuring the rotation maintains their connection.
    
    Args:
        center_x (float): x-coordinate of the center of the parallelogram.
        center_y (float): y-coordinate of the center of the parallelogram.
        radius (float): Length of the shortest edge of the isosceles triangles.
        angle (float): Rotation angle of the parallelogram.
        size (int): Size of the image.
        
    Returns:
        image (2D np.array): The resulting image with the parallelogram filled.
    """
    # Initialize the image with zeros
    image = np.zeros((size, size), dtype=np.float64)

    # Define the rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    # Define the center of the first triangle
    center1 = np.array([center_x, center_y])

    # Define the vertices of the first triangle (angles in radians)
    theta1 = np.array([90 * np.pi / 180, 180 * np.pi / 180, 270 * np.pi / 180]) + np.pi / 2
    vertices1 = np.array([
        center1[0] + radius * np.cos(theta1),
        center1[1] + radius * np.sin(theta1)
    ]).T

    # Apply rotation to the first triangle
    vertices1_rotated = (rotation_matrix @ (vertices1 - center1).T).T + center1

    # Draw the first triangle
    rr1, cc1 = draw.polygon(vertices1_rotated[:, 1], vertices1_rotated[:, 0], image.shape)
    image[rr1, cc1] = 0.62

    # Define the center of the second triangle
    offset = np.array([radius, -radius])  # Offset by the shortest edges
    center2 = center1 + offset

    # Define the vertices of the second triangle (angles in radians)
    theta2 = np.array([90 * np.pi / 180, 0, 270 * np.pi / 180]) + np.pi / 2
    vertices2 = np.array([
        center2[0] + radius * np.cos(theta2),
        center2[1] + radius * np.sin(theta2)
    ]).T

    # Apply rotation to the second triangle (around the shared center of rotation)
    vertices2_rotated = (rotation_matrix @ (vertices2 - center1).T).T + center1

    # Draw the second triangle
    rr2, cc2 = draw.polygon(vertices2_rotated[:, 1], vertices2_rotated[:, 0], image.shape)
    
    image[rr2, cc2] = 0.62

    return image


# === Dummy placeholder functions for triangle/pentagon ===
def load_data(name, folder_path="simulation_model"):
    # Initialize a list to store the loaded data
    synthetic_data = []

    # Loop through all files in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the file is a .npy file and contains 'triangle' in its name
        if filename.endswith(".npy") and name in filename:
            # Construct the full path to the file
            file_path = os.path.join(folder_path, filename)
            # Load the .npy file and append it to the list
            data = np.load(file_path)
            synthetic_data.append(data)

    return synthetic_data


# Define function to create rotated and shifted triangle overlays
def transform_data(data_mask, background_mask, angle=0, shift_val=(0, 0)):
    # Rotate and shift the triangle mask
    rotated_data = rotate(data_mask.astype(float), angle, reshape=False, order=0)
    shifted_data = shift(rotated_data, shift_val, order=0)

    # Create a new array with the rotated and shifted triangle over the background
    transformed_array = np.where(shifted_data >= 0.5, 0.62, background_mask)
    return transformed_array


# === Main generator ===
def generate_and_save_data(folder_path):
    cfg = get_config()
    size, num_samples = cfg['size'], cfg['num_samples']
    background_mask = np.zeros((size, size), dtype=np.float64)

    rotation_label = np.random.randint(0, 36, num_samples)
    num = np.random.randint(0, 8, num_samples)
    shift1 = np.random.randint(-10, 10, num_samples)
    shift2 = np.random.randint(-10, 10, num_samples)
    radius_range = np.random.randint(int(90 * cfg['image_scale']), int(110 * cfg['image_scale']), num_samples)

    rotation_range = cfg['rotation_range']

    # Triangle
    P_t, L_t = [], []
    tri_data = load_data('triangle', folder_path)
    for i in range(num_samples):
        angle = rotation_range[rotation_label[i]]
        img = transform_data(tri_data[num[i]], background_mask, angle, (shift1[i], shift2[i]))
        P_t.append(img)
        L_t.append(angle)

    # Pentagon
    P_f, L_f = [], []
    pen_data = load_data('pentagon', folder_path)
    for i in range(num_samples):
        angle = rotation_range[rotation_label[i]]
        img = transform_data(pen_data[num[i]], background_mask, angle, (shift1[i], shift2[i]))
        P_f.append(img)
        L_f.append(angle)

    # Parallelogram
    P_fo, L_fo = [], []
    for i in range(num_samples):
        angle = rotation_range[rotation_label[i]]
        center_x = int(size / 2) + shift1[i]
        center_y = int(size / 2) + shift2[i]
        radius = radius_range[i]
        img = create_parallelogram(center_x, center_y, radius, angle, size)
        P_fo.append(img)
        L_fo.append(angle)

    # Save all
    data = {
        'triangles': (P_t, L_t),
        'pentagons': (P_f, L_f),
        'parallelograms': (P_fo, L_fo)
    }
    return P_t, L_t, P_f, L_f, P_fo, L_fo

P_t, L_t, P_f, L_f, P_fo, L_fo = generate_and_save_data("simulation_model")