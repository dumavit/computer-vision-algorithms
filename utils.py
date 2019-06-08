import numpy as np

from PIL import Image as pil_image


def zero_padding(im, padding_size):
    # Wrap image with zero rows and columns
    height, width, channels = im.shape

    hzeros = np.zeros((width, padding_size, channels))
    vzeros = np.zeros((padding_size, height + 2 * padding_size, channels))

    return np.vstack((vzeros, np.hstack((hzeros, im, hzeros)), vzeros))


def convolve_image(im, kernel):
    # Apply convolution for every image channel. 
    # Use zero_padding to preserve image shape.
    if im.ndim == 2:
        im = im.reshape((im.shape[0], im.shape[1], 1))
    height, width, channels = im.shape
    kernel_size, _ = kernel.shape
    padding = kernel_size // 2

    image_padded = zero_padding(im, padding)
    result = np.zeros(im.shape)

    for channel in range(channels):
        for i in range(width):
            for j in range(height):
                patch = image_padded[i : i + kernel_size, j : j + kernel_size, channel]
                result[i, j, channel] = np.sum(np.multiply(kernel, patch))

    return result

