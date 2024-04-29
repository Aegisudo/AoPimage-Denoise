import numpy as np
from scipy.interpolate import Rbf  # Radial basis function interpolator
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from PIL import Image


def find_extrema(data):
    # Identifying local maxima and minima
    local_max = argrelextrema(data, np.greater, order=10, mode='wrap')
    local_min = argrelextrema(data, np.less, order=10, mode='wrap')
    return local_max, local_min


def interpolate_surface(coords, values, data_shape):
    # Interpolate using radial basis function
    x, y = np.meshgrid(np.arange(data_shape[1]), np.arange(data_shape[0]))
    rbf = Rbf(coords[1], coords[0], values, function='thin_plate')
    return rbf(y, x)


def is_monotonic(data):
    # Checks if the data has minimal extrema, suggesting it's close to monotonic
    maxima, minima = find_extrema(data)
    if len(maxima[0]) < 5 and len(minima[0]) < 5:  # Adjust threshold as necessary
        return True
    return False


def bemd(image):
    imfs = []
    residual = image.copy()
    while True:
        maxima, minima = find_extrema(residual)
        if len(maxima[0]) == 0 and len(minima[0]) == 0:
            break

        maxima_values = residual[maxima]
        minima_values = residual[minima]

        max_surface = interpolate_surface(maxima, maxima_values, residual.shape)
        min_surface = interpolate_surface(minima, minima_values, residual.shape)

        mean_surface = (max_surface + min_surface) / 2
        imf = residual - mean_surface
        residual = residual - imf

        imfs.append(imf)

        # Check if residual is monotonic or has few extrema
        if is_monotonic(residual):
            break

        # Plotting for visualization
        plt.imshow(imf, cmap='gray')
        plt.title(f'Extracted IMF {len(imfs)}')
        plt.colorbar()
        plt.show()

    return imfs, residual


# Load an existing image
image_path = 'aop.png'  # Update this with the actual path to your image
image = Image.open(image_path)
image = image.convert('L')  # Convert the image to grayscale
image = image.resize((400, 400))  # Ensure the image is 400x400
image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]

imfs, residual = bemd(image_array)

plt.imshow(residual, cmap='gray')
plt.title('Final Residual')
plt.colorbar()
plt.show()
