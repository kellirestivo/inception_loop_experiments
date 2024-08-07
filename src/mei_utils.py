import numpy as np
import torch
from tqdm import tqdm
import datajoint as dj
from itertools import product
from scipy.ndimage import shift
from scipy.ndimage import center_of_mass

from natural_images_utils import rescale_image

from nnfabrik.main import *
import nnfabrik
from nnfabrik import main, builder

from mei import mixins
from mei import legacy

import nnvision
from nnvision.tables.main import Recording
from nnvision.tables.from_mei import MEI
from nnvision.utility.experiment_helpers.mei_masks import generate_mask
from nnvision.utility.experiment_helpers.image_processing import get_norm, re_norm

schema = CustomSchema(dj.config.get('nnfabrik.schema_name', 'nnfabrik_core'))

if not 'stores' in dj.config:
    dj.config['stores'] = {}
dj.config['stores'] = {
  'minio': {
    'protocol': 'file',
    'location': '/mnt/dj-stor01/',
    'stage': '/mnt/dj-stor01/'
  }
}


@schema
class RFCLMaskedControl(dj.Manual):
    definition = """
    -> StaticImage.Image
    cl_image_id          : int
    ---
    image_mask           : longblob                   # mask from MEI used for controls
    image_mask_key       : longblob
    image                : longblob
    """

@schema
class RFShiftCLMaskedMEI(dj.Manual):
    definition = """
    cl_image_id          : int
    method_fn            : varchar(64)                  # name of the method function
    method_hash          : varchar(32)                  # hash of the method config
    dataset_fn           : varchar(64)                  # name of the dataset loader function
    dataset_hash         : varchar(64)                  # hash of the configuration object
    ensemble_hash        : char(32)                     # the hash of the ensemble
    data_key             : varchar(64)                  # 
    unit_id              : int                          # 
    unit_type            : int                          # 
    mei_seed             : tinyint unsigned             # MEI seed
    ---
    image_mask           : longblob
    shift                : longblob
    image                : longblob
    """

def fetch_mei_from_db(mei_key):
    """
    Fetch MEI from the database using the provided key.

    Args:
        mei_key (dict): The key to query the MEI.

    Returns:
        numpy.ndarray: The fetched MEI if found, None otherwise.
    """

    try:
        if len(MEI & mei_key) == 1:
            mei = (MEI & mei_key).fetch1("mei")
            mei = torch.load(mei).detach().cpu().numpy().squeeze()[0, ...]  # shape: (2,80,80), channels, h, w
            return mei
        elif len(MEI & mei_key) > 1:
            print("Warning - Found too many MEI for this unit")
        else:
            print("Warning - No MEIs found")
    except Exception as e:
        print(f"Error fetching MEI: {e}")
    return None

def get_unit_index(unit_key, dataset_hash):
    """
    Retrieve the unit index from the Recording.Units table.

    Args:
        unit_key (dict): The key to query the unit.
        dataset_hash (str): The dataset hash.

    Returns:
        int: The unit index if found, None otherwise.
    """
    try:
        unit_index = (Recording.Units & dict(unit_id=unit_key['unit_id'], dataset_hash=dataset_hash)).fetch1('unit_index')
        return unit_index
    except Exception as e:
        print(f"Error fetching unit index: {e}")
        return None
    
def process_meis(selected_keys, seed_list, ensemble_list, method_hash, dataset_hash, data_key):
    """
    Process MEIs for each model ensemble and model seed.

    Args:
        selected_keys (list): List of selected unit keys.
        seed_list (list): List of MEI seeds.
        ensemble_list (list): List of ensemble hashes.
        method_hash (str): Method hash.
        dataset_hash (str): Dataset hash.
        data_key (str): Data key.

    Returns:
        dict: Dictionary containing the processed MEIs.
    """

    MEI_dictionary = {i: {} for i in ensemble_list}
    for ensemble_hash in ensemble_list:
        for unit_key in tqdm(selected_keys):
            MEI_list = []
            for seed in seed_list:
                key = dict(ensemble_hash=ensemble_hash, mei_seed=seed, method_hash=method_hash)
                mei_key = dj.AndList([unit_key, key])
                mei = fetch_mei_from_db(mei_key)
                if mei:
                    unit_index = get_unit_index(unit_key, dataset_hash)
                    if unit_index:
                        MEI_list.append(
                            dict(image=mei,
                                 unit_index=unit_index,
                                 ensemble_hash=ensemble_hash,
                                 datajoint_key=dict(ensemble_hash=ensemble_hash,
                                                    dataset_hash=dataset_hash,
                                                    data_key=data_key,
                                                    unit_id=unit_key['unit_id'],
                                                    mei_seed=seed)
                                 )
                        )
            MEI_dictionary[ensemble_hash][unit_index] = MEI_list
    return MEI_dictionary


def initialize_mei_mask_dictionary(ensemble_list, selected_indices):
    """
    Initialize the MEI mask dictionary.

    Args:
        ensemble_list (list): List of ensemble hashes.
        selected_indices (list): List of selected unit indices.

    Returns:
        dict: Initialized MEI mask dictionary.
    """
    return {ensemble: {unit_index: {} for unit_index in selected_indices} for ensemble in ensemble_list}

def generate_masks_for_unit(MEI_dictionary, ensemble, unit_index, zscore_thresh, gaussian_sigma):
    """
    Generate masks for a specific unit in the ensemble.

    Args:
        MEI_dictionary (dict): Dictionary containing MEIs.
        ensemble (str): Ensemble hash.
        unit_index (int): Unit index.
        zscore_thresh (float): Z-score threshold for mask generation.
        gaussian_sigma (float): Gaussian sigma for mask generation.

    Returns:
        list: List of generated masks.
    """
    return [
        generate_mask(i['image'], zscore_thresh=zscore_thresh, gaussian_sigma=gaussian_sigma)
        for i in MEI_dictionary[ensemble][unit_index]
    ]

def get_percentage_of_clipping(mask, threshold=.25):
    """
    Get percentage of MEI mask that is clipped off based on z-score threshold

    Args:
        mask (np.ndarray): MEI mask
        threshold (float): z-score threshold 
    Returns:
        float: percentage of mask that is clipped
    """
    a = (mask[:, -1:] > threshold).sum() / mask[:, -1:].size
    b = (mask[:, :1] > threshold).sum() / mask[:, :1].size
    c = (mask[:1, :] > threshold).sum() / mask[:1, :].size
    d = (mask[-1:, :] > threshold).sum() / mask[-1:, :].size
    return(a + b + c +d) / 4


def get_non_clipped_masks_indices(masks, threshold=0.1):
    """
    Get indices of non-clipped masks.

    Args:
        masks (list): List of masks.
        threshold (float): Clipping threshold.

    Returns:
        list: Indices of non-clipped masks.
    """
    percentage_clipped = [get_percentage_of_clipping(mask) for mask in masks]
    return [i for i, percentage in enumerate(percentage_clipped) if percentage < threshold]


def process_unit(ensemble, unit_index, MEI_dictionary, param_config, mei_mask_dictionary):
    """
    Process a single unit in the ensemble.

    Args:
        ensemble (str): Ensemble hash.
        unit_index (int): Unit index.
        MEI_dictionary (dict): Dictionary containing MEIs.
        param_config (dict): Parameter configuration.
        mei_mask_dictionary (dict): Dictionary to store processed results.

    Returns:
        None
   """
    masks = generate_masks_for_unit(MEI_dictionary, ensemble, unit_index, param_config['zscore_thresh'], param_config['gaussian_sigma'])
    non_clipped_masks_indices = get_non_clipped_masks_indices(masks)

    if len(non_clipped_masks_indices) < 3:
        print(f'Fewer than 2 masks made the cut for unit: {unit_index}')
        non_clipped_masks_indices = list(np.argsort([get_percentage_of_clipping(mask) for mask in masks])[:3])

    unmasked_norms = [get_norm(i['image']).item() for i in MEI_dictionary[ensemble][unit_index]]
    masked_meis = [
        re_norm(masks[i] * MEI_dictionary[ensemble][unit_index][i]['image'], param_config['final_image_norm']).cpu().numpy().squeeze(0)
        for i in range(len(MEI_dictionary[ensemble][unit_index]))
    ]
    masked_norms = [get_norm(i).item() for i in masked_meis]
    datajoint_keys = [i['datajoint_key'] for i in MEI_dictionary[ensemble][unit_index]]

    mei_mask_dictionary[ensemble][unit_index].update({
        'nonclipped_masked_meis': [masked_meis[i] for i in non_clipped_masks_indices],
        'nonclipped_masks': [masks[i] for i in non_clipped_masks_indices],
        'nonclipped_masked_norms': [masked_norms[i] for i in non_clipped_masks_indices],
        'nonclipped_unmasked_norms': [unmasked_norms[i] for i in non_clipped_masks_indices],
        'masks': masks,
        'nonclipped_masks_indices': non_clipped_masks_indices,
        'unmasked_norms': unmasked_norms,
        'masked_meis': masked_meis,
        'masked_norms': masked_norms,
        'datajoint_keys': datajoint_keys
    })

def process_all_units(ensemble_list, selected_indices, MEI_dictionary, param_config):
    """
    Process all units in all ensembles.

    Args:
        ensemble_list (list): List of ensemble hashes.
        selected_indices (list): List of selected unit indices.
        MEI_dictionary (dict): Dictionary containing MEIs.
        param_config (dict): Parameter configuration.

    Returns:
        dict: Dictionary containing processed MEI masks.
    """
    mei_mask_dictionary = initialize_mei_mask_dictionary(ensemble_list, selected_indices)
    for ensemble in ensemble_list:
        for unit_index in tqdm(selected_indices):
            process_unit(ensemble, unit_index, MEI_dictionary, param_config, mei_mask_dictionary)
    return mei_mask_dictionary


def get_all_shifts(mei, start, end, step):
    x = np.arange(start, end, step)
    meis = np.zeros((len(x)**2, *mei.shape[-2:]))
    shifts = np.zeros((len(x)**2, 2))
    if len(mei.shape) != 2:
        mei = mei.squeeze()
    if len(mei.shape) != 2:
        raise ValueError("the mei must be either a 2d array or a 4d array of size (1, 1, h, w)")
    
    for i, (x_i, y_i) in enumerate(product(x,x)):
        meis[i] = shift(mei, (x_i, y_i))
        shifts[i] = np.array([x_i, y_i])
    return meis, shifts

def get_best_shifted_mei(mei, masks, unit_index, gauss_validator_ensemble, data_key, models):
    """
    Get the best shifted MEI and corresponding response.

    Args:
        mei (torch.Tensor): The MEI tensor.
        masks (torch.Tensor): The masks tensor.
        unit_index (int): The unit index.
        gauss_validator_ensemble (str): The gauss validator ensemble hash.
        data_key (str): The data key.
        models (dict): Dictionary of models.

    Returns:
        tuple: Best shifted MEI, best shifted response, best shifted mask.
    """
    shifted_meis, all_shifts = get_all_shifts(mei.squeeze(), -30, 30, 2)
    shifted_meis = torch.from_numpy(shifted_meis).to(torch.float32).unsqueeze(1)
    
    with torch.no_grad():
        output = models[gauss_validator_ensemble](shifted_meis.cuda(), data_key=data_key)[..., unit_index].cpu()

    best_shifted_mei_index = np.argmax(output)
    best_shifted_mei_response = np.max(output.numpy())
    best_shifted_mask = shift(masks, all_shifts[best_shifted_mei_index])
    best_masked_mei = shifted_meis[best_shifted_mei_index].squeeze(0)
    
    return best_masked_mei, best_shifted_mei_response, best_shifted_mask

def process_unit_best_meis(ensemble, unit_index, mei_mask_dictionary, gauss_validator_ensemble, data_key, models):
    """
    Process the best MEIs for a single unit in the ensemble.

    Args:
        ensemble (str): Ensemble hash.
        unit_index (int): Unit index.
        mei_mask_dictionary (dict): Dictionary containing MEI masks.
        gauss_validator_ensemble (str): The gauss validator ensemble hash.
        data_key (str): The data key.
        models (dict): Dictionary of models.

    Returns:
        dict: Dictionary with best MEI and mask information.
    """
    best_masked_meis_unit = []
    best_masks_unit = []
    best_shifted_masked_meis_response = []
    
    meis = torch.from_numpy(np.array(mei_mask_dictionary[ensemble][unit_index]['nonclipped_masked_meis']))
    masks = torch.from_numpy(np.array(mei_mask_dictionary[ensemble][unit_index]['nonclipped_masks']))
    
    all_outputs = []
    for mei, mask in zip(meis, masks):
        best_masked_mei, best_shifted_mei_response, best_shifted_mask = get_best_shifted_mei(mei, mask, unit_index, gauss_validator_ensemble, data_key, models)
        best_shifted_masked_meis_response.append(best_shifted_mei_response)
        best_masked_meis_unit.append(best_masked_mei)
        best_masks_unit.append(best_shifted_mask)
        all_outputs.append(best_shifted_mei_response)
    
    best_mei_response_index = np.argmax(all_outputs)
    
    return {
        'best_shifted_masked_meis_responses': all_outputs,
        'best_shifted_masked_meis': best_masked_meis_unit,
        'best_shifted_masks': best_masks_unit,
        'single_best_masked_mei': best_masked_meis_unit[best_mei_response_index],
        'single_best_mask': best_masks_unit[best_mei_response_index],
        'single_best_dj_key': mei_mask_dictionary[ensemble][unit_index]["datajoint_keys"][best_mei_response_index]
    }

def process_all_units_best_meis(ensemble_list, selected_indices, mei_mask_dictionary, gauss_validator_ensemble, data_key, models):
    """
    Process the best MEIs for all units in all ensembles.

    Args:
        ensemble_list (list): List of ensemble hashes.
        selected_indices (list): List of selected unit indices.
        mei_mask_dictionary (dict): Dictionary containing MEI masks.
        gauss_validator_ensemble (str): The gauss validator ensemble hash.
        data_key (str): The data key.
        models (dict): Dictionary of models.

    Returns:
        dict: Updated MEI mask dictionary with best MEI and mask information.
    """
    for ensemble in ensemble_list:
        for unit_index in tqdm(selected_indices):
            best_meis_info = process_unit_best_meis(ensemble, unit_index, mei_mask_dictionary, gauss_validator_ensemble, data_key, models)
            mei_mask_dictionary[ensemble][unit_index].update(best_meis_info)
    return mei_mask_dictionary

def process_for_presentation(img, monitor_scale_factor=11.4, img_mean=124.34, img_std=70.28):
    """
    Process the best MEIs for all units in all ensembles.

    Args:
        img (np.ndarray): experiment image to be processed
        monitor_scale_factor (float): factor to scale image up to monitor dimensions
        img_mean (float): mean of images in training set
        img_std (float): std of images in training set

    Returns:
        np.ndarray: final processed image
    """
    # upscale, convert, clip, change type
    upsampled_image = rescale_image(img, monitor_scale_factor)

    #Convert image to pixel space
    convertedImage = upsampled_image * img_std + img_mean

    # Convert image to 8-bit
    finalImage = np.round(convertedImage).clip(0,255).astype('uint8')

    return finalImage