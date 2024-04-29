import numpy as np
from scipy.interpolate import Rbf  # Radial basis function interpolator
from scipy.signal import argrelextrema
from scipy.stats import entropy
import matplotlib.pyplot as plt
from PIL import Image


def find_extrema(data):
    # Identifying local maxima and minima using a larger neighborhood for a larger image
    local_max = argrelextrema(data, np.greater, order=10, mode='wrap')
    local_min = argrelextrema(data, np.less, order=10, mode='wrap')
    return local_max, local_min


def interpolate_surface(coords, values, data_shape):
    # Interpolate using radial basis function
    x, y = np.meshgrid(np.arange(data_shape[1]), np.arange(data_shape[0]))
    rbf = Rbf(coords[1], coords[0], values, function='thin_plate')
    return rbf(y, x)


def compute_entropy(data):
    # Compute the histogram of the data
    histogram, _ = np.histogram(data, bins=256, range=(0, 1))
    # Normalize the histogram
    histogram_normalized = histogram / histogram.sum()
    # Compute the entropy
    return entropy(histogram_normalized)


def bemd(image, num_imfs=10):
    imfs = []
    residual = image.copy()
    entropies = []
    for i in range(num_imfs):
        maxima, minima = find_extrema(residual)
        maxima_values = residual[maxima]
        minima_values = residual[minima]

        max_surface = interpolate_surface(maxima, maxima_values, residual.shape)
        min_surface = interpolate_surface(minima, minima_values, residual.shape)

        mean_surface = (max_surface + min_surface) / 2
        imf = residual - mean_surface
        residual = residual - imf

        imfs.append(imf)

        # Compute entropy for the IMF
        imf_entropy = compute_entropy(imf.flatten())
        entropies.append(imf_entropy)

        # Plotting for visualization
        plt.imshow(imf, cmap='gray')
        plt.title(f'IMF {i + 1}')
        plt.colorbar()
        plt.show()

    # Compute entropy for the final residual
    residual_entropy = compute_entropy(residual.flatten())
    entropies.append(residual_entropy)

    # Plotting entropy values
    plt.figure(figsize=(8, 4))
    plt.plot(entropies, marker='o', linestyle='-')
    plt.xlabel('Component Number (IMFs and Residual)')
    plt.ylabel('Entropy')
    plt.title('Entropy of IMFs and Residual')
    plt.grid(True)
    plt.show()

    return imfs, residual


# Load an existing image
image_path = 'aop1.png'  # Update this with the actual path to your image
image = Image.open(image_path)
image = image.convert('L')  # Convert the image to grayscale
image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]

imfs, residual = bemd(image_array)

plt.imshow(residual, cmap='gray')
plt.title('Residual')
plt.colorbar()
plt.show()
