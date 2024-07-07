import numpy as np
import math
import logging
from scipy.ndimage import shift as nd_shift

def shift_based_on_rf(mei, rf_size, increment=0.3, ppd=5.7, padsize=0, num_shifts=19, num_circ_segments = 6, com_h=None, com_w=None):
    """
    Shifts the MEI in a pre-specified grid based on the size of the neuron's receptive field.

    Args:
        mei (array): The MEI itself as a numpy array, with the shape of (2, h, w) = (transparency, height, width).
        rf_size (float): The receptive field size of this session.
        increment (float): The amplitude of the radius of the RF shift. Default is 0.3.
        ppd (float): Pixels per degree, needed for the conversion of RF size into pixels. Default is 5.7.
        padsize (int): Padding size before the shift. Default is 0.
        num_shifts (int): Total budgeted number of shifts to shift the image.
        num_circ_segments (int): Number of circle segments to shift the image inside.
        com_h (float, optional): Center of mass height. Default is None.
        com_w (float, optional): Center of mass width. Default is None.

    Returns:
        shifted_images (array): The shifted images.
        shifts (array): The corresponding shifts for the images.
    """

    shifted_images = np.ones((num_shifts, mei.shape[1] + padsize * 2, mei.shape[2] + padsize * 2))
    shifts = np.zeros((num_shifts, 2))

    if len(mei.shape) == 3:
        mei = np.stack([np.pad(mei[0], padsize), np.pad(mei[1], padsize)])
        img = mei[0]
        h_orig, w_orig = center_of_mass(mei[1])
    else:
        mei = np.stack([np.pad(mei, padsize)])
        img = mei
        h_orig, w_orig = com_h, com_w

    shifts[0, :] = (h_orig, w_orig)
    shifted_images[0] = img

    img_count = 0
    for k in range(1, 4):
        amp = rf_size * increment * ppd * k
        circ_segs = (np.arange(num_circ_segments) / num_circ_segments * 2 * math.pi) + k * math.pi / num_circ_segments
        for x in circ_segs:
            img_count += 1
            h_new = h_orig + np.cos(x) * amp
            w_new = w_orig + np.sin(x) * amp
            shifts[img_count, :] = (h_new, w_new)
            shifted_images[img_count] = shift(img, (h_new - h_orig, w_new - w_orig))

    return shifted_images, shifts


def center_of_mass(image):
    """
    Computes the center of mass of the given image.
    
    Args:
        image (array): The input image.
    
    Returns:
        tuple: (height, width) coordinates of the center of mass.
    """
    h, w = np.indices(image.shape)
    total = image.sum()
    com_h = (h * image).sum() / total
    com_w = (w * image).sum() / total
    return com_h, com_w


def shift(image, shift_values):
    """
    Shifts the image by the specified values.
    
    Args:
        image (array): The input image.
        shift_values (tuple): (height_shift, width_shift) values.
    
    Returns:
        array: The shifted image.
    """
    from scipy.ndimage import shift as nd_shift
    return nd_shift(image, shift_values)


