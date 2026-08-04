import json
import os
import pandas as pd
import numpy as np

# Load metadata
with open('metadata_submetered.json', 'r') as f:
    metadata = json.load(f)

# Exclude light bulbs / lamps to focus on other appliances
EXCLUDE_TYPES = ['Compact Fluorescent Lamp', 'Incandescent Light Bulb', 'Fluorescent Lamp']

def extract_features(df):
    """Calculate key electrical features from raw signal samples."""
    current = df['Current'].values
    voltage = df['Voltage'].values
    
    rms_current = np.sqrt(np.mean(current**2))
    rms_voltage = np.sqrt(np.mean(voltage**2))
    peak_current = np.max(np.abs(current))
    apparent_power = rms_voltage * rms_current
    
    return {
        "rms_current": round(float(rms_current), 3),
        "rms_voltage": round(float(rms_voltage), 3),
        "peak_current": round(float(peak_current), 3),
        "apparent_power": round(float(apparent_power), 3)
    }

processed_samples = []

# Handle dict or list metadata formats
items = metadata.items() if isinstance(metadata, dict) else enumerate(metadata)

for sample_id, item in items:
    # Handle structure where sample_id comes from dict key or item dict
    if isinstance(metadata, list):
        sample_id = item.get('id', sample_id)

    app_info = item.get('appliance', {})
    if isinstance(app_info, dict):
        app_type = app_info.get('type', '')
    else:
        app_type = str(app_info)

    # Skip light bulbs
    if app_type in EXCLUDE_TYPES or not app_type:
        continue

    # Try finding the CSV file in submetered/submetered_new folder
    csv_filename = f"{sample_id}.csv"
    csv_path = os.path.join('submetered', 'submetered_new', csv_filename)

    # Fallback check if CSVs are directly in submetered or current directory
    if not os.path.exists(csv_path):
        if os.path.exists(os.path.join('submetered', csv_filename)):
            csv_path = os.path.join('submetered', csv_filename)
        elif os.path.exists(csv_filename):
            csv_path = csv_filename

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, header=None, names=['Current', 'Voltage'])
            
            features = extract_features(df.iloc[:500])
            features['appliance'] = app_type
            features['sample_id'] = str(sample_id)
            
            processed_samples.append(features)
        except Exception as e:
            continue

# Convert to DataFrame
df_features = pd.DataFrame(processed_samples)

print(f"Total processed samples (excluding lamps): {len(df_features)}")
if not df_features.empty:
    print("\n--- Appliances Found ---")
    print(df_features['appliance'].value_counts())
    print("\n--- Feature Preview ---")
    print(df_features.head(10))
else:
    print("No matching files found. Check your folder structure.")