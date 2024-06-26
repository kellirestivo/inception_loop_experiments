import os
import yaml
from src.input_handling import get_filename
from src.data_file_processing import process_mat_file
from src.data_saving import save_as_pickle

def load_config(config_file='pickle_file_paths.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def preprocess_data(config):
    PATH = '/mnt/lab/users/kelli/mei/V4_SUA_closed_loop'
    pickle_paths = config['paths']['pickle_paths']
    variable_names = [
        'subject_id', 'session_id', 'training_image_ids', 'training_prior_image_ids',
        'training_responses', 'testing_image_ids', 'testing_prior_image_ids',
        'testing_responses', 'testing_repetitions', 'unit_ids', 'electrode_nums', 
        'x_grid_location', 'y_grid_location', 'relative_depth_microns', 'rf_size'
    ]

        # Generate filename based on the current date
    todays_file = get_filename()

    # Process each .mat file in the directory
    for filename in os.listdir(PATH):
        if filename == f"{todays_file}.mat":
            session_dict = process_mat_file(PATH, filename, variable_names)
            if session_dict:
                root_filename = filename.split(".mat")[0]
                save_as_pickle(session_dict, root_filename, pickle_paths)

if __name__ == "__main__":
    config = load_config()
    preprocess_data(config)