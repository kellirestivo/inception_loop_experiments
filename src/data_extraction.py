import os
import scipy.io as sio

def process_mat_file(path, filename, variable_names):
    """
    Load and process a .mat file.

    Args:
        path (str): Directory path where .mat file is located.
        filename (str): Name of the .mat file.
        variable_names (list): List of variable names to extract from the .mat file.

    Returns:
        dict: Dictionary containing extracted variables.
    """

    try:
        path_session = os.path.join(path, filename)
        print(f"Processing file: {path_session}")

        # Load .mat dataset file of interest
        session = sio.loadmat(path_session, squeeze_me=True)

        # Convert data to dictionary
        session_values = session['sess_data'].item()
        session_dict = {variable_names[i]: session_values[i] for i in range(len(session_values))}
        return session_dict
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return None
    