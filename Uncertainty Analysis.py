# ==============================================================================
# TabPFN Prediction Uncertainty Analysis Script (Outputs: Mean, SD, CI) - Final Version
# ==============================================================================
# This script executes a complete workflow:
# 1. Loads the training data.
# 2. Reads and processes the 12 feature TIF files for national-scale prediction.
# 3. Runs a bootstrapping loop to train multiple TabPFN models.
# 4. Calculates the final mean prediction, standard deviation (SD), and 95% confidence interval width (CI).
# 5. Saves these three results as separate TIF files.
#
# Instructions for use:
# 1. Ensure all required libraries are installed:
#    pip install pandas numpy scikit-learn tabpfn rasterio
# 2. The configuration section below has been set up based on your provided paths.
# 3. Place this script in an environment with the required libraries and execute it.
# ==============================================================================

import os
import time
import numpy as np
import pandas as pd
import rasterio
from sklearn.utils import resample
from tabpfn import TabPFNRegressor

# ==============================================================================
# --- 1. Configuration Section - Set up according to your environment ---
# ==============================================================================

# --- Path to the training data CSV file ---
TRAINING_CSV_PATH = r"C:/Users/13600K/Desktop/中国预测/全部变量-晒完特征 - 改名.csv"

# --- Column name of the target variable (dependent variable) in the CSV file ---
TARGET_COLUMN_NAME = "HCO3"

# --- Folder path containing the 12 feature TIF files ---
FEATURE_TIF_FOLDER = r"C:/Users/13600K/Desktop/中国预测/中国变量-xin"

# --- List of feature names in the correct order ---
FEATURE_NAMES_ORDERED = [
    'PS', 'PET', 'TS', 'PCQ', 'GCC', 'GWSA',
    'TWI', 'DNC', 'CSC', 'LAI', 'SOC', 'SP'
]

# --- Save path for the output TIF files ---
OUTPUT_FOLDER = r"C:/Users/13600K/Desktop/中国预测"

# --- Bootstrapping parameters ---
N_BOOTSTRAPS = 20


# ==============================================================================
# --- Helper Functions (Do Not Modify) ---
# ==============================================================================

def process_feature_tifs(tif_folder, feature_names):
    """
    Reads, stacks, and processes a series of feature TIF files, converting them
    into a NumPy array for prediction and handling NoData values.
    """
    print("--- Reading and processing feature TIF files ---")
    tif_files = [os.path.join(tif_folder, f"{name}.tif") for name in feature_names]

    # Check if all files exist
    for f_path in tif_files:
        if not os.path.exists(f_path):
            raise FileNotFoundError(f"ERROR: Required TIF file not found: {f_path}")
    print(f"Successfully found all {len(tif_files)} TIF files.")

    all_features_list = []
    profile = None      # To store raster metadata
    nodata_value = None # To store the NoData value
    height, width = 0, 0  # To store raster dimensions

    for i, tif_path in enumerate(tif_files):
        print(f"  Reading feature {i + 1}/{len(tif_files)}: {os.path.basename(tif_path)}")
        with rasterio.open(tif_path) as src:
            if profile is None:
                profile = src.profile
                nodata_value = src.nodata
                height, width = src.height, src.width
                print(f"  Raster properties set: {width}x{height} pixels.")
                if nodata_value is None:
                    print("  Warning: NoData value not found in TIF metadata.")
                else:
                    print(f"  Detected NoData value: {nodata_value}")

            feature_array = src.read(1).flatten()
            all_features_list.append(feature_array)

    print("\n  Stacking all features into a single large array...")
    x_to_predict = np.vstack(all_features_list).T
    print(f"  Initial grid shape: {x_to_predict.shape} (total pixels, number of features)")

    print("  Identifying valid pixels (handling NoData)...")
    if nodata_value is not None:
        # A pixel is invalid if any feature has the NoData value
        valid_mask = ~np.any(x_to_predict == nodata_value, axis=1)
        x_to_predict_valid = x_to_predict[valid_mask, :]
        valid_indices = np.where(valid_mask)[0]
    else:
        # If no NoData value is defined, assume all pixels are valid
        valid_mask = np.ones(x_to_predict.shape[0], dtype=bool)
        x_to_predict_valid = x_to_predict
        valid_indices = np.arange(x_to_predict.shape[0])

    print(f"  Total pixels: {x_to_predict.shape[0]}")
    print(f"  Valid pixels for prediction: {x_to_predict_valid.shape[0]}")

    metadata = {
        'profile': profile,
        'height': height,
        'width': width,
        'nodata_value': nodata_value if nodata_value is not None else -9999,
        'valid_indices': valid_indices
    }
    return x_to_predict_valid, metadata


def save_array_as_tif(data_array_1d, metadata, output_filename):
    """Saves a 1D NumPy array back to a TIF file using reference metadata."""
    print(f"\n--- Saving result to TIF: {os.path.basename(output_filename)} ---")

    # Create a full-sized array filled with the NoData value
    full_array_flat = np.full(metadata['height'] * metadata['width'],
                              metadata['nodata_value'],
                              dtype=np.float32)

    # Place the valid prediction data into the correct positions
    full_array_flat[metadata['valid_indices']] = data_array_1d

    # Reshape the flat array to the original 2D raster dimensions
    output_raster = full_array_flat.reshape(metadata['height'], metadata['width'])

    # Update the metadata profile for the output file
    profile = metadata['profile']
    profile.update(dtype=rasterio.float32, count=1, nodata=metadata['nodata_value'])

    # Write the data to a new TIF file
    with rasterio.open(output_filename, 'w', **profile) as dst:
        dst.write(output_raster, 1)

    print(f"  Successfully saved to: {output_filename}")


# ==============================================================================
# --- Main Script Execution ---
# ==============================================================================

if __name__ == "__main__":
    start_time = time.time()
    print("==========================================================")
    print("      Starting TabPFN Uncertainty Analysis Workflow (SD & CI)")
    print("==========================================================")

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}")

    print("\n--- Step 1: Loading training data ---")
    try:
        train_df_full = pd.read_csv(TRAINING_CSV_PATH, encoding='utf-8')
        train_df_full.columns = train_df_full.columns.str.strip() # Clean up column names
        # Validate that all required columns exist
        assert TARGET_COLUMN_NAME in train_df_full.columns
        for feature in FEATURE_NAMES_ORDERED:
            assert feature in train_df_full.columns
        # Select only the necessary columns in the correct order
        train_df = train_df_full[FEATURE_NAMES_ORDERED + [TARGET_COLUMN_NAME]]
        print(f"Successfully loaded training data with shape: {train_df.shape}")
    except Exception as e:
        print(f"ERROR: Failed to load or validate training data. Please check the path and column names. Details: {e}")
        exit()

    print("\n--- Step 2: Processing prediction grid TIF files ---")
    try:
        X_to_predict_valid, grid_metadata = process_feature_tifs(FEATURE_TIF_FOLDER, FEATURE_NAMES_ORDERED)
    except Exception as e:
        print(f"ERROR: Failed to process TIF files. Please check paths and file integrity. Details: {e}")
        exit()

    print("\n--- Step 3: Starting bootstrapping loop ---")
    all_predictions = []

    for i in range(N_BOOTSTRAPS):
        loop_start_time = time.time()
        print(f"  > Processing bootstrap model {i + 1}/{N_BOOTSTRAPS}...")

        # Create a bootstrap sample (sampling with replacement)
        boot_sample = resample(train_df, n_samples=len(train_df), replace=True, random_state=i)
        X_boot = boot_sample[FEATURE_NAMES_ORDERED]
        y_boot = boot_sample[TARGET_COLUMN_NAME]

        # ---[Final Correction Point]---
        # Add ignore_pretraining_limits=True to allow processing >1000 samples on CPU
        model_boot = TabPFNRegressor(device='cpu', ignore_pretraining_limits=True)

        # Fit the model on the bootstrap sample
        model_boot.fit(X_boot, y_boot)

        # Make predictions on the valid geospatial data
        predictions_boot = model_boot.predict(X_to_predict_valid)
        all_predictions.append(predictions_boot)

        loop_end_time = time.time()
        print(f"    ... iteration completed in {loop_end_time - loop_start_time:.2f} seconds.")

    print("\n  Bootstrapping loop completed.")

    print("\n--- Step 4: Calculating final statistical metrics ---")
    all_predictions_np = np.array(all_predictions)

    print("  Calculating mean prediction...")
    final_prediction_mean = np.mean(all_predictions_np, axis=0)

    print("  Calculating standard deviation (SD)...")
    uncertainty_sd = np.std(all_predictions_np, axis=0)

    print("  Calculating 95% confidence interval width (CI)...")
    # CI Width = 97.5th percentile - 2.5th percentile
    lower_bound = np.percentile(all_predictions_np, 2.5, axis=0)
    upper_bound = np.percentile(all_predictions_np, 97.5, axis=0)
    uncertainty_ci_width = upper_bound - lower_bound

    print("  Metrics calculation complete.")

    print("\n--- Step 5: Saving all results as TIF files ---")
    save_array_as_tif(final_prediction_mean, grid_metadata, os.path.join(OUTPUT_FOLDER, 'prediction_mean.tif'))
    save_array_as_tif(uncertainty_sd, grid_metadata, os.path.join(OUTPUT_FOLDER, 'uncertainty_sd.tif'))
    save_array_as_tif(uncertainty_ci_width, grid_metadata, os.path.join(OUTPUT_FOLDER, 'uncertainty_ci95_width.tif'))

    end_time = time.time()
    print("\n==========================================================")
    print("      Workflow completed successfully!")
    print(f"      Total execution time: {(end_time - start_time) / 60:.2f} minutes.")
    print(f"      Output files have been saved in: {OUTPUT_FOLDER}")
    print("      - prediction_mean.tif (Mean prediction map)")
    print("      - uncertainty_sd.tif (Standard deviation uncertainty map)")
    print("      - uncertainty_ci95_width.tif (95% confidence interval width map)")
    print("==========================================================")