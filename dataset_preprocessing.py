import os
import yaml
from src.input_handling import get_filename
from src.data_extraction import process_mat_file
from src.data_saving import save_as_pickle

def load_config(config_file='pickle_file_paths.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def preprocess_data(
        config, 
        filepath='/mnt/lab/users/kelli/mei/V4_SUA_closed_loop',
        variable_names = [
        'subject_id', 'session_id', 'training_image_ids', 'training_prior_image_ids',
        'training_responses', 'testing_image_ids', 'testing_prior_image_ids',
        'testing_responses', 'testing_repetitions', 'unit_ids', 'electrode_nums', 
        'x_grid_location', 'y_grid_location', 'relative_depth_microns', 'rf_size'
    ]):
    """
    Extract monkey ephys dataset from .mat file and save to .pickle file

    Args:
        config: .yaml file of paths to save dataset
        filepath (str): path to dataset
        variable_names (list): list of variable names to extract from the .mat file.
    Returns:
        None
    """

    pickle_paths = config['paths']['pickle_paths']

    # Input today's filename
    todays_file = get_filename()

    # Process each .mat file in the directory
    for filename in os.listdir(filepath):
        if filename == f"{todays_file}.mat":
            session_dict = process_mat_file(filepath, filename, variable_names)
            if session_dict:
                root_filename = filename.split(".mat")[0]
                save_as_pickle(session_dict, root_filename, pickle_paths)

if __name__ == "__main__":
    config = load_config()
    preprocess_data(config)