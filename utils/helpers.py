# utils/helpers.py
import yaml
import os
import pandas as pd

def load_config(config_path='config.yaml'): # Changed default to config.yaml in project root
    """Loads configuration from a YAML file."""
    print(f"DEBUG: helpers.py - Loading config from: {os.path.abspath(config_path)}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path_or_filepath):
    """
    Ensures that a directory exists. If a filepath is provided,
    it ensures the parent directory of that filepath exists.
    """
    if os.path.isdir(path_or_filepath):
        directory_to_create = path_or_filepath
    else:
        directory_to_create = os.path.dirname(path_or_filepath)

    if directory_to_create and not os.path.exists(directory_to_create):
        os.makedirs(directory_to_create, exist_ok=True)
        print(f"DEBUG: helpers.py - Created directory: {os.path.abspath(directory_to_create)}")

def load_data(filepath):
    """Loads data from a CSV file."""
    print(f"DEBUG: helpers.py - Attempting to load data from: {os.path.abspath(filepath)}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")
    return pd.read_csv(filepath)

def save_data(df, filepath):
    """Saves a DataFrame to a CSV file, ensuring the parent directory exists."""
    print(f"DEBUG: helpers.py - Attempting to save data to: {os.path.abspath(filepath)}")
    ensure_dir(filepath) # Ensure the parent directory for the output file exists
    df.to_csv(filepath, index=False)
    print(f"DEBUG: helpers.py - Data saved to {os.path.abspath(filepath)}")
