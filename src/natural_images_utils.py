import torch
from tqdm import tqdm
import numpy as np
import nnvision
from nnvision.utility.experiment_helpers.image_processing import re_norm

def initialize_natural_images_dictionary(ensemble, selected_indices):
    """
    Initialize the natural images dictionary.

    Parameters:
    ensemble (str): Ensemble hash.
    selected_indices (list): List of selected unit indices.

    Returns:
    dict: Initialized natural images dictionary.
    """
    return {ensemble: {unit_index: {} for unit_index in selected_indices}}

def apply_mask_to_images(transformed_images, masks, h, w, norm_value=19):
    """
    Apply masks to all transformed images and renormalize them.

    Parameters:
    transformed_images (list): List of transformed images.
    masks (torch.Tensor): Tensor of masks.
    h (int): Height of the images.
    w (int): Width of the images.
    norm_value (int): Normalization value.

    Returns:
    torch.Tensor: Tensor of masked and renormalized images.
    """
    num_images = len(transformed_images)
    num_masks = len(masks)
    masked_images = torch.empty((num_images, num_masks, h, w))
    
    for i, mask in enumerate(masks):
        for j, image in enumerate(transformed_images):
            masked_images[j, i, ...] = re_norm(image['image'].squeeze(0) * mask, norm_value)
    
    return masked_images

def create_datajoint_keys(masks, transformed_images, meiMaskDictionary, ensemble, unit_index):
    """
    Create DataJoint keys for each masked image.

    Parameters:
    masks (torch.Tensor): Tensor of masks.
    transformed_images (list): List of transformed images.
    meiMaskDictionary (dict): Dictionary containing MEI masks.
    ensemble (str): Ensemble hash.
    unit_index (int): Unit index.

    Returns:
    list: List of DataJoint keys.
    """
    keys = []
    selected_keys = [meiMaskDictionary[ensemble][unit_index]['datajoint_keys'][i] for i in range(len(masks))]
    
    for i, mask in enumerate(masks):
        for j, image in enumerate(transformed_images):
            this_key = {k: v for (k, v) in selected_keys[i].items()}
            this_key['image_id'] = image['image_id']
            this_key['image_class'] = image['image_class']
            keys.append(this_key)
    
    return keys

def process_natural_images_for_unit(unit_index, transformed_images, meiMaskDictionary, ensemble, h, w, norm_value=19):
    """
    Process natural images for a single unit.

    Parameters:
    unit_index (int): Unit index.
    transformed_images (list): List of transformed images.
    meiMaskDictionary (dict): Dictionary containing MEI masks.
    ensemble (str): Ensemble hash.
    h (int): Height of the images.
    w (int): Width of the images.
    norm_value (int): Normalization value.

    Returns:
    dict: Dictionary containing masked images and DataJoint keys.
    """
    masks = torch.from_numpy(np.array(meiMaskDictionary[ensemble][unit_index]['best_shifted_masks']))
    masked_images = apply_mask_to_images(transformed_images, masks, h, w, norm_value)
    datajoint_keys = create_datajoint_keys(masks, transformed_images, meiMaskDictionary, ensemble, unit_index)
    
    return {
        'images': masked_images,
        'datajoint_keys': datajoint_keys
    }

def process_all_units_natural_images(ensemble, selected_indices, transformed_images, meiMaskDictionary, h, w, norm_value=19):
    """
    Process natural images for all units in an ensemble.

    Parameters:
    ensemble (str): Ensemble hash.
    selected_indices (list): List of selected unit indices.
    transformed_images (list): List of transformed images.
    meiMaskDictionary (dict): Dictionary containing MEI masks.
    h (int): Height of the images.
    w (int): Width of the images.
    norm_value (int): Normalization value.

    Returns:
    dict: Updated natural images dictionary with masked images and DataJoint keys.
    """
    natural_images_dict = initialize_natural_images_dictionary(ensemble, selected_indices)
    
    for unit_index in tqdm(selected_indices):
        processed_data = process_natural_images_for_unit(unit_index, transformed_images, meiMaskDictionary, ensemble, h, w, norm_value)
        natural_images_dict[ensemble][unit_index].update(processed_data)
    
    return natural_images_dict