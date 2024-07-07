import torch
from tqdm import tqdm
import numpy as np
from skimage.transform import rescale

import nnvision
from nnvision.utility.experiment_helpers.image_processing import re_norm


def rescale_image(image, scale):
    """
    Rescales the image by the specified scale factor; adapted from the transformations defined within the
    ImageCache class; makes use of skimage.transform.rescale to interpolate the image

    Args:
        image (np.ndarray): input image
        scale (float or tuple of floats): specified scale factor


        mode (str): what to do to values outside the boundaries of the input
        multichannel (bool): whether the last axis of the image should be interpreted as channels or spatial dimension
        anti_aliasing (bool): whether or not to Gaussian blur the image before interpolation to avoid aliasing artifacts
        preserve_range (bool): whether to keep original range of values
    Returns:
        np.ndarray: rescaled image
    """

    rescale_fn = lambda x, s: rescale(
        x,
        s,
        mode="reflect",
        multichannel=False,
        anti_aliasing=False,
        preserve_range=True,
    ).astype(x.dtype)

    image = image if scale == 1 else rescale_fn(image, scale)

    return image
def initialize_natural_images_dictionary(ensemble, selected_indices):
    """
    Initialize the natural images dictionary.

    Args:
        ensemble (str): Ensemble hash.
        selected_indices (list): List of selected unit indices.

    Returns:
        dict: Initialized natural images dictionary.
    """
    return {ensemble: {unit_index: {} for unit_index in selected_indices}}

def apply_mask_to_images(transformed_images, masks, h, w, norm_value=19):
    """
    Apply masks to all transformed images and renormalize them.

    Args:
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

def create_datajoint_keys(masks, transformed_images, mei_mask_dictionary, ensemble, unit_index):
    """
    Create DataJoint keys for each masked image.

    Args:
        masks (torch.Tensor): Tensor of masks.
        transformed_images (list): List of transformed images.
        mei_mask_dictionary (dict): Dictionary containing MEI masks.
        ensemble (str): Ensemble hash.
        unit_index (int): Unit index.

    Returns:
        list: List of DataJoint keys.
    """
    keys = []
    selected_keys = [mei_mask_dictionary[ensemble][unit_index]['datajoint_keys'][i] for i in range(len(masks))]
    
    for i, mask in enumerate(masks):
        for j, image in enumerate(transformed_images):
            this_key = {k: v for (k, v) in selected_keys[i].items()}
            this_key['image_id'] = image['image_id']
            this_key['image_class'] = image['image_class']
            keys.append(this_key)
    
    return keys

def process_natural_images_for_unit(unit_index, transformed_images, mei_mask_dictionary, ensemble, h, w, norm_value=19):
    """
    Process natural images for a single unit.

    Args:
        unit_index (int): Unit index.
        transformed_images (list): List of transformed images.
        mei_mask_dictionary (dict): Dictionary containing MEI masks.
        ensemble (str): Ensemble hash.
        h (int): Height of the images.
        w (int): Width of the images.
        norm_value (int): Normalization value.

    Returns:
        dict: Dictionary containing masked images and DataJoint keys.
    """
    masks = torch.from_numpy(np.array(mei_mask_dictionary[ensemble][unit_index]['best_shifted_masks']))
    masked_images = apply_mask_to_images(transformed_images, masks, h, w, norm_value)
    datajoint_keys = create_datajoint_keys(masks, transformed_images, mei_mask_dictionary, ensemble, unit_index)
    
    return {
        'images': masked_images,
        'datajoint_keys': datajoint_keys
    }

def process_all_units_natural_images(ensemble, selected_indices, transformed_images, mei_mask_dictionary, h, w, norm_value=19):
    """
    Process natural images for all units in an ensemble.

    Args:
        ensemble (str): Ensemble hash.
        selected_indices (list): List of selected unit indices.
        transformed_images (list): List of transformed images.
        mei_mask_dictionary (dict): Dictionary containing MEI masks.
        h (int): Height of the images.
        w (int): Width of the images.
        norm_value (int): Normalization value.

    Returns:
        dict: Updated natural images dictionary with masked images and DataJoint keys.
    """
    natural_images_dict = initialize_natural_images_dictionary(ensemble, selected_indices)
    
    for unit_index in tqdm(selected_indices):
        processed_data = process_natural_images_for_unit(unit_index, transformed_images, mei_mask_dictionary, ensemble, h, w, norm_value)
        natural_images_dict[ensemble][unit_index].update(processed_data)
    
    return natural_images_dict



def initialize_masked_renormed_image_dict(ensemble, selected_indices):
    """
    Initialize the masked renormed image dictionary.

    Args:
        ensemble (str): Ensemble hash.
        selected_indices (list): List of selected unit indices.

    Returns:
        dict: Initialized masked renormed image dictionary.
    """
    return {ensemble: {unit_index: {} for unit_index in selected_indices}}

def process_masked_images(unit_index, masks, masked_renormed_natural_images, models, ensemble, data_key):
    """
    Process masked images for a single unit.

    Args:
        unit_index (int): Unit index.
        masks (np.array): Array of masks.
        masked_renormed_natural_images (torch.Tensor): Tensor of masked renormed natural images.
        models (dict): Dictionary of models.
        ensemble (str): Ensemble hash.
        data_key (str): The data key.

    Returns:
        dict: Dictionary containing best masks, best MEI masks, max responses, and all responses.
    """
    n_total_natural_images, h, w = masked_renormed_natural_images.shape[0], masked_renormed_natural_images.shape[2], masked_renormed_natural_images.shape[3]
    best_masks = np.zeros((n_total_natural_images, h, w))
    best_mei_masks = np.zeros((n_total_natural_images, h, w))
    max_responses = np.zeros((n_total_natural_images))
    all_natural_image_responses = np.zeros((n_total_natural_images, len(masks)))

    for mask_index in range(len(masks)):
        responses = []
        for batch in range(0, 5000, 1000):
            with torch.no_grad():
                responses.append(models[ensemble](masked_renormed_natural_images[batch:batch+1000, mask_index, ...].unsqueeze(1).cuda(), data_key=data_key)[:, unit_index])
        responses = torch.cat(responses).cpu().numpy()
        updated_best_response_indices = np.where(responses > max_responses)[0]
        max_responses[updated_best_response_indices] = responses[updated_best_response_indices]
        best_mei_masks[updated_best_response_indices] = masks[mask_index]
        best_masks[updated_best_response_indices] = masked_renormed_natural_images[updated_best_response_indices, mask_index, ...]
        all_natural_image_responses[:, mask_index] = responses

    return best_masks, best_mei_masks, max_responses, all_natural_image_responses

def select_top_images(max_responses, best_masks, best_mei_masks, datajoint_keys_best_mask, n_best_images_selected):
    """
    Select the top images based on the maximum responses.

    Args:
        max_responses (np.array): Array of maximum responses.
        best_masks (np.array): Array of best masks.
        best_mei_masks (np.array): Array of best MEI masks.
        datajoint_keys_best_mask (np.array): Array of DataJoint keys.
        n_best_images_selected (int): Number of best images to select.

    Returns:
        dict: Dictionary containing top masks, top responses, top MEI masks, and selected DataJoint keys.
    """
    best_image_argsort = np.flipud(np.argsort(max_responses))[:n_best_images_selected]
    top_responses = max_responses[best_image_argsort]
    top_masks = best_masks[best_image_argsort]
    top_mei_masks = best_mei_masks[best_image_argsort]
    selected_datajoint_keys = datajoint_keys_best_mask[best_image_argsort]

    return {
        'images_best_mask': top_masks,
        'responses_best_mask': top_responses,
        'mei_masks': top_mei_masks,
        'datajoint_keys': selected_datajoint_keys
    }

def process_all_units_masked_images(ensemble, selected_indices, transformed_images, mei_mask_dictionary, natural_images_dictionary, models, data_key, n_best_images_selected):
    """
    Process masked images for all units in an ensemble.

    Args:
        ensemble (str): Ensemble hash.
        selected_indices (list): List of selected unit indices.
        transformed_images (list): List of transformed images.
        mei_mask_dictionary (dict): Dictionary containing MEI masks.
        natural_images_dictionary (dict): Dictionary containing natural images.
        models (dict): Dictionary of models.
        data_key (str): The data key.
        n_best_images_selected (int): Number of best images to select.

    Returns:
        dict: Updated masked renormed image dictionary with best masks and responses.
    """
    masked_renormed_image_dict = initialize_masked_renormed_image_dict(ensemble, selected_indices)
    
    n_total_natural_images = len(transformed_images)
    h, w = transformed_images[0]['image'].shape[-2], transformed_images[0]['image'].shape[-1]

    for unit_index in tqdm(selected_indices):
        masks = mei_mask_dictionary[ensemble][unit_index]['best_shifted_masks']
        masked_renormed_natural_images = natural_images_dictionary[ensemble][unit_index]['images']
        all_datajoint_keys = natural_images_dictionary[ensemble][unit_index]['datajoint_keys']
        datajoint_keys_best_mask = np.array(all_datajoint_keys[0:5000])

        best_masks, best_mei_masks, max_responses, all_natural_image_responses = process_masked_images(
            unit_index, masks, masked_renormed_natural_images, models, ensemble, data_key
        )

        top_images_data = select_top_images(max_responses, best_masks, best_mei_masks, datajoint_keys_best_mask, n_best_images_selected)
        
        masked_renormed_image_dict[ensemble][unit_index].update({
            'images_best_mask': top_images_data['images_best_mask'],
            'responses_best_mask': top_images_data['responses_best_mask'],
            'responses_all_masks': all_natural_image_responses,
            'mei_masks': top_images_data['mei_masks'],
            'datajoint_keys': top_images_data['datajoint_keys']
        })
    
    return masked_renormed_image_dict