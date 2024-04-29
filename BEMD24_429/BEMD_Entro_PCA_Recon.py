import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import Rbf
from scipy.signal import argrelextrema
from scipy.stats import entropy
from sklearn.decomposition import PCA

def find_extrema(data):
    local_max = argrelextrema(data, np.greater, order=10, mode='wrap')
    local_min = argrelextrema(data, np.less, order=10, mode='wrap')
    return local_max, local_min

def interpolate_surface(coords, values, data_shape):
    x, y = np.meshgrid(np.arange(data_shape[1]), np.arange(data_shape[0]))
    rbf = Rbf(coords[1], coords[0], values, function='thin_plate')
    return rbf(y, x)

def compute_histogram_entropy(data):
    histogram, _ = np.histogram(data, bins=256, range=(0, 1))
    histogram_normalized = histogram / (histogram.sum() + np.finfo(float).eps)
    return entropy(histogram_normalized)

def bemd(image, num_imfs=10):
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
        if len(imfs) >= num_imfs:
            break
    return imfs, residual

def pca_denoise(imfs):
    denoised_imfs = []
    pca_params = []
    for imf in imfs:
        data = imf.flatten().reshape(-1, 1)
        pca = PCA()
        pca.fit(data)
        retained_components = pca.explained_variance_ratio_ > 0.1
        retained_data = pca.transform(data)[:, retained_components]
        reconstructed = pca.inverse_transform(np.pad(retained_data,
                                                     ((0, 0), (0, pca.n_components_ - retained_data.shape[1])),
                                                     'constant', constant_values=0))
        denoised_imf = reconstructed.reshape(imf.shape)
        denoised_imfs.append(denoised_imf)
        pca_params.append(pca.explained_variance_ratio_)
    return denoised_imfs, pca_params

def reconstruct_image(imfs):
    return np.sum(imfs, axis=0)

# Load and process an existing image
image_path = 'aop.png'
image = Image.open(image_path)
image = image.convert('L')
image_array = np.array(image) / 255.0

# BEMD to extract IMFs and the final residual
imfs, final_residual = bemd(image_array, num_imfs=10)

# PCA denoising on the first 5 IMFs
denoised_imfs, pca_params = pca_denoise(imfs[:5])

# Compute entropy for original and denoised IMFs
original_entropies = [compute_histogram_entropy(imf) for imf in imfs] + [compute_histogram_entropy(final_residual)]
denoised_entropies = [compute_histogram_entropy(imf) for imf in denoised_imfs]

# Save all entropies and PCA contribution ratios to a file
np.savez('imf_data.npz', original_entropies=original_entropies, denoised_entropies=denoised_entropies, pca_params=pca_params)

# Visualization of all IMFs, the original image, and the final residual
cols = 4
rows = (len(imfs) + 2) // cols + 1  # +1 for original and residual

plt.figure(figsize=(cols * 4, rows * 4))
plt.subplot(rows, cols, 1)
plt.imshow(image_array, cmap='gray')
plt.title('Original Image')
plt.colorbar()

for idx, imf in enumerate(imfs + [final_residual], 2):
    plt.subplot(rows, cols, idx)
    plt.imshow(imf, cmap='gray')
    plt.title(f'Component {idx - 1}')
    plt.colorbar()

plt.tight_layout()
plt.show()

# Plotting original entropies with annotations
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(original_entropies)+1), original_entropies, '-o', label='Original Entropies (IMFs + Residual)')
for i, v in enumerate(original_entropies):
    plt.text(i + 1, v, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
plt.xticks(range(1, len(original_entropies)+1))
plt.xlabel('Component Index')
plt.ylabel('Entropy')
plt.title('Entropy of Original IMFs and Residual')
plt.grid(True)
plt.legend()
plt.show()

# Plotting denoised entropies with annotations
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(denoised_entropies)+1), denoised_entropies, '-o', label='Denoised IMFs Entropy')
for i, v in enumerate(denoised_entropies):
    plt.text(i + 1, v, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
plt.xticks(range(1, len(denoised_entropies)+1))
plt.xlabel('Denoised IMF Index')
plt.ylabel('Entropy')
plt.title('Entropy of Denoised IMFs')
plt.grid(True)
plt.legend()
plt.show()

# Reconstruct and visualize the denoised image from the first five denoised IMFs
reconstructed_image = reconstruct_image(denoised_imfs)
plt.figure(figsize=(8, 8))
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image from Denoised IMFs')
plt.colorbar()
plt.show()
