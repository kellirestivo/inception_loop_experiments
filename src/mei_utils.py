import numpy as np
import torch
from tqdm import tqdm
import datajoint as dj

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