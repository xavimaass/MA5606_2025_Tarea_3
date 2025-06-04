import numpy as np
import torch

# Función para generar datos reales (2D Concentric Circles)
def sample_concentric_circles(batch_size, num_circles=2, noise=0.01):
    """
    Generates data points forming concentric circles.

    Args:
        batch_size (int): The number of samples to generate.
        num_circles (int): The number of concentric circles.
        noise (float): The amount of noise to add to the radius.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 2) containing the generated points.
    """
    data = []
    samples_per_circle = batch_size // num_circles
    remaining_samples = batch_size % num_circles

    for i in range(num_circles):
        current_samples = samples_per_circle + (1 if i < remaining_samples else 0)
        radius = (i + 1.0) / num_circles  # Radii from 1/num_circles to 1
        theta = 2 * np.pi * np.random.rand(current_samples)
        r = radius + noise * np.random.randn(current_samples)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        data.append(np.vstack([x, y]).T)

    data = np.vstack(data)
    np.random.shuffle(data) # Shuffle to mix samples from different circles
    return torch.tensor(data, dtype=torch.float32)


def sample_s_curve(batch_size, noise=0.01):
    """
    Generates data points forming an S-curve.

    Args:
        batch_size (int): The number of samples to generate.
        noise (float): The amount of noise to add to the y-coordinates.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 2) containing the generated points.
    """
    # Generate x values uniformly distributed
    x = np.random.rand(batch_size) * 4 * np.pi - 2 * np.pi # Range from -2pi to 2pi

    # Generate y values using a sine function to create the S-shape
    y = np.sin(x)

    # Add noise to the y values
    y += noise * np.random.randn(batch_size)

    # Stack x and y to form the 2D dataset
    data = np.vstack([x, y]).T

    # Shuffle the data (optional, but good practice for training)
    np.random.shuffle(data)

    return torch.tensor(data, dtype=torch.float32)