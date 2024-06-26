import numpy as np
import torch
from tqdm import tqdm
import datajoint as dj

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


def fetch_mei_from_db(mei_key):
    """
    Fetch MEI from the database using the provided key.

    Parameters:
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

    Parameters:
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

    Parameters:
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

    Parameters:
    ensemble_list (list): List of ensemble hashes.
    selected_indices (list): List of selected unit indices.

    Returns:
    dict: Initialized MEI mask dictionary.
    """
    return {ensemble: {unit_index: {} for unit_index in selected_indices} for ensemble in ensemble_list}

def generate_masks_for_unit(MEI_dictionary, ensemble, unit_index, zscore_thresh, gaussian_sigma):
    """
    Generate masks for a specific unit in the ensemble.

    Parameters:
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
    a = (mask[:, -1:] > threshold).sum() / mask[:, -1:].size
    b = (mask[:, :1] > threshold).sum() / mask[:, :1].size
    c = (mask[:1, :] > threshold).sum() / mask[:1, :].size
    d = (mask[-1:, :] > threshold).sum() / mask[-1:, :].size
    return(a + b + c +d) / 4


def get_non_clipped_masks_indices(masks, threshold=0.1):
    """
    Get indices of non-clipped masks.

    Parameters:
    masks (list): List of masks.
    threshold (float): Clipping threshold.

    Returns:
    list: Indices of non-clipped masks.
    """
    percentage_clipped = [get_percentage_of_clipping(mask) for mask in masks]
    return [i for i, percentage in enumerate(percentage_clipped) if percentage < threshold]


def process_unit(ensemble, unit_index, MEI_dictionary, param_config, meiMaskDictionary):
    """
    Process a single unit in the ensemble.

    Parameters:
    ensemble (str): Ensemble hash.
    unit_index (int): Unit index.
    MEI_dictionary (dict): Dictionary containing MEIs.
    param_config (dict): Parameter configuration.
    meiMaskDictionary (dict): Dictionary to store processed results.

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

    meiMaskDictionary[ensemble][unit_index].update({
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

    Parameters:
    ensemble_list (list): List of ensemble hashes.
    selected_indices (list): List of selected unit indices.
    MEI_dictionary (dict): Dictionary containing MEIs.
    param_config (dict): Parameter configuration.

    Returns:
    dict: Dictionary containing processed MEI masks.
    """
    meiMaskDictionary = initialize_mei_mask_dictionary(ensemble_list, selected_indices)
    for ensemble in ensemble_list:
        for unit_index in tqdm(selected_indices):
            process_unit(ensemble, unit_index, MEI_dictionary, param_config, meiMaskDictionary)
    return meiMaskDictionary