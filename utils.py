import os
import json
import pickle as pkl
import shutil

def open_file(filepath):
    """Open a file and return its contents as a string."""
    with open(filepath) as f:
        return f.read()
    
def load_json(filepath):
    """Load a JSON file and return its contents as a dictionary."""
    with open(filepath) as f:
        return json.load(f)
    
def save_json(js, target):
    """Save data as a JSON file."""
    with open(target, 'w') as f:
        json.dump(js, f, indent=2)

def generate_save_path(save_path, id, ext, file_type="frames"):
    """
    Generate a save path for a file with organized directory structure.
    
    Args:
        save_path (str): Base directory to save files
        id (str): File identifier
        ext (str): File extension
        file_type (str): Type of file (for naming)
        
    Returns:
        str: Full save path for the file
    """
    tmp = id[:4]
    save_dir = os.path.join(save_path, tmp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if file_type:
        save_path = os.path.join(save_dir, f"{id}_{file_type}.{ext}")
    return save_path

def save_arr_to_pkl(data, save_path, id, file_type="frames"):
    """
    Save data to a pickle file with organized directory structure.
    
    Args:
        data: Data to save
        save_path (str): Base directory to save files
        id (str): File identifier
        file_type (str): Type of file (for naming)
    """
    path = generate_save_path(
        save_path, id, "pkl", file_type)
    save_to_pkl(data, path)

def save_to_pkl(data, path):
    with open(path, "wb") as f:
        pkl.dump(data, f)

        
def print_matrix(m):
    for r in m:
        print(r)