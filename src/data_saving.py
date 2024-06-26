import os
import pickle

def save_as_pickle(session_dict, root_filename, pickle_paths):
    """
    Save the processed data as pickle files in the specified paths.

    Parameters:
    session_dict (dict): Dictionary containing the session data.
    root_filename (str): The root filename for the pickle files.
    pickle_paths (list): List of paths to save the pickle files.
    """
    for path in pickle_paths:
        try:
            print(f"Saving to path: {path}")
            with open(os.path.join(path, f"{root_filename}.pickle"), 'wb') as f:
                pickle.dump(session_dict, f)
        except Exception as e:
            print(f"Error saving file to {path}: {e}")