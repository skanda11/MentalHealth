# main.py
import os
import sys # For debug output if needed
from utils.helpers import load_config, ensure_dir
from preprocessing.preprocess import preprocess_data
from classification.train_model import train_classification_model
from classification.predict import predict_distress
from temporal_analysis.analyze_trends import analyze_temporal_trends

# DEBUG: Print when the main module is first loaded
print("DEBUG: main.py - Module imported and starting execution path.")

def run_pipeline(config_path='config.yaml'): # Adjusted default to config.yaml in project root
    print(f"\nDEBUG: main.py - Entering run_pipeline function (config_path: {config_path}).")
    
    # Load configuration
    print("DEBUG: main.py - Attempting to load configuration...")
    config = load_config(config_path)
    print("DEBUG: main.py - Configuration loaded successfully.")
    
    # Extract raw data path from config
    raw_data_filepath = config['data']['raw_data_path'] # Adjusted key from 'preprocessing' to 'data'
    print(f"DEBUG: main.py - Raw data file path from config: '{raw_data_filepath}'")

    # --- Step 0: Verify Local Data Exists ---
    # This crucial check determines if the local CSV is in place.
    print(f"DEBUG: main.py - Checking for existence of local raw data file: '{raw_data_filepath}'...")
    if not os.path.exists(raw_data_filepath):
        print("="*80, file=sys.stderr) # Print to stderr for emphasis
        print(f"FATAL ERROR: The local data file was not found at the expected path.", file=sys.stderr)
        print(f"Expected Path: {raw_data_filepath}", file=sys.stderr)
        print("Please ensure you have manually downloaded 'Suicide_Detection.csv' and placed it in the 'data/raw/' directory.", file=sys.stderr)
        print("Pipeline aborted.", file=sys.stderr)
        print("="*80, file=sys.stderr)
        return  # Exit the pipeline cleanly if the file is truly missing

    print(f"DEBUG: main.py - Local raw data file found at '{raw_data_filepath}'. Proceeding.")
    print("\n--- Starting Pipeline with Local Data ---")

    # --- Step 1: Preprocessing ---
    print("\n--- Starting Preprocessing ---")
    print("DEBUG: main.py - Invoking preprocess_data function...")
    preprocess_data(config['data']) # Pass 'data' config section
    print("DEBUG: main.py - preprocess_data function completed.")

    # --- Step 2: Model Training ---
    print("\n--- Starting Model Training ---")
    print("DEBUG: main.py - Invoking train_classification_model function...")
    train_classification_model(config) # Pass full config
    print("DEBUG: main.py - train_classification_model function completed.")

    # --- Step 3: Prediction ---
    print("\n--- Starting Prediction ---")
    print("DEBUG: main.py - Invoking predict_distress function...")
    predict_distress(config) # Pass full config
    print("DEBUG: main.py - predict_distress function completed.")

    # --- Step 4: Temporal Analysis ---
    print("\n--- Starting Temporal Analysis ---")
    print("DEBUG: main.py - Invoking analyze_temporal_trends function...")
    analyze_temporal_trends(config) # Pass full config
    print("DEBUG: main.py - analyze_temporal_trends function completed.")

    print("\n--- Pipeline Execution Complete ---")
    print("DEBUG: main.py - All main pipeline steps finished successfully.")
    print("To view the dashboard, navigate to the project root and run: `streamlit run dashboard/app.py`")
    print("DEBUG: main.py - Exiting run_pipeline function.")


if __name__ == '__main__':
    print("\nDEBUG: main.py - Script started, running __main__ block.")
    
    # Ensure necessary output directories exist
    print("DEBUG: main.py - Ensuring output directories exist...")
    ensure_dir("data/processed")
    ensure_dir("data/predictions")
    ensure_dir("models/classifier")
    
    # Also ensure the base 'data' directory for raw data exists.
    # We explicitly ensure 'data/raw' is created if the user manually places data there.
    ensure_dir("data/raw") 
    print("DEBUG: main.py - Output directories checked/created.")
    
    run_pipeline()
    print("DEBUG: main.py - Script finished executing.")
