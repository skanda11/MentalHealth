# temporal_analysis/analyze_trends.py
import pandas as pd
from utils.helpers import load_data, save_data # Adjusted import path

print("DEBUG: analyze_trends.py - Module loaded.")

def analyze_temporal_trends(config): # Accepts full config
    print("DEBUG: analyze_trends.py - Entering analyze_temporal_trends function.")
    class_config = config['classification']
    temp_config = config['temporal_analysis']
    
    input_path = class_config['predictions_output_path']
    output_path = temp_config['output_trends_path']

    print(f"DEBUG: analyze_trends.py - Loading predicted data from {input_path}...")
    df = load_data(input_path)
    print(f"DEBUG: analyze_trends.py - Data loaded. Shape: {df.shape}")

    df = df.sort_index().reset_index(drop=True)
    print("DEBUG: analyze_trends.py - Data sorted for temporal analysis.")
    
    df['smoothed_trend'] = df['distress_probability'].rolling(
        window=temp_config['moving_average_period'], min_periods=1
    ).mean()
    print(f"DEBUG: analyze_trends.py - Smoothed trend calculated with window size {temp_config['moving_average_period']}.")
    
    df['is_alert'] = (df['smoothed_trend'] >= temp_config['alert_threshold']).astype(int)
    print(f"DEBUG: analyze_trends.py - 'is_alert' flag set based on threshold {temp_config['alert_threshold']}.")
    
    s = df['is_alert'].groupby((df['is_alert'] != df['is_alert'].shift()).cumsum()).cumsum()
    df['triggered_alert'] = (s >= temp_config['consecutive_alerts_required']).astype(int)
    print(f"DEBUG: analyze_trends.py - 'triggered_alert' identified for {temp_config['consecutive_alerts_required']} consecutive alerts.")

    save_data(df, output_path)
    print(f"DEBUG: analyze_trends.py - Temporal analysis results saved to {output_path}.")
    print("DEBUG: analyze_trends.py - Exiting analyze_temporal_trends function.")
