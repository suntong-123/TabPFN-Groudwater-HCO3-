import os
import time
import numpy as np
import pandas as pd
import rasterio
from sklearn.utils import resample
from tabpfn import TabPFNRegressor

# ==============================================================================
# --- 1. CONFIGURATION AREA ---
# ==============================================================================

# --- Path to the training data CSV file ---
TRAINING_CSV_PATH = r"C:/Users/13600K/Desktop/中国预测/地质云-1051个点-全部变量 8.29-晒完特征 - 改名.csv"

# --- Target variable (dependent variable) column name ---
TARGET_COLUMN_NAME = "HCO3"

# --- Folder path containing the 12 feature TIF files ---
FEATURE_TIF_FOLDER = r"C:/Users/13600K/Desktop/中国预测/中国变量-xin"

# --- Feature names in the specific order ---
FEATURE_NAMES_ORDERED = [
    'PS', 'PET', 'TS', 'PCQ', 'GCC', 'GWSA',
    'TWI', 'DNC', 'CSC', 'LAI', 'SOC', 'SP'
]

# --- Output settings ---
OUTPUT_FOLDER = r"C:/Users/13600K/Desktop/中国预测/Sensitivity_Analysis"
# Directory for temporary prediction files (supports resuming after interruption)
TEMP_DIR = os.path.join(OUTPUT_FOLDER, "temp_predictions")

# --- Bootstrapping parameters ---
N_BOOTSTRAPS_TOTAL = 100  # Total iterations set to 100
COMPARISON_POINTS = [30, 50, 100]  # Output results at these milestones


# ==============================================================================
# --- 2. HELPER FUNCTIONS ---
# ==============================================================================

def process_feature_tifs(tif_folder, feature_names):
    print("--- Reading and processing feature TIF files ---")
    tif_files = [os.path.join(tif_folder, f"{name}.tif") for name in feature_names]
    for f_path in tif_files:
        if not os.path.exists(f_path):
            raise FileNotFoundError(f"Error: Required TIF file not found: {f_path}")

    all_features_list = []
    profile = None
    nodata_value = None
    height, width = 0, 0

    for i, tif_path in enumerate(tif_files):
        print(f"  Reading feature {i + 1}/{len(tif_files)}: {os.path.basename(tif_path)}")
        with rasterio.open(tif_path) as src:
            if profile is None:
                profile = src.profile
                nodata_value = src.nodata
                height, width = src.height, src.width
            feature_array = src.read(1).flatten()
            all_features_list.append(feature_array)

    x_to_predict = np.vstack(all_features_list).T

    # Create valid data mask based on NoData value
    if nodata_value is not None:
        valid_mask = ~np.any(x_to_predict == nodata_value, axis=1)
        x_to_predict_valid = x_to_predict[valid_mask, :]
        valid_indices = np.where(valid_mask)[0]
    else:
        valid_indices = np.arange(x_to_predict.shape[0])
        x_to_predict_valid = x_to_predict

    metadata = {
        'profile': profile,
        'height': height,
        'width': width,
        'nodata_value': nodata_value if nodata_value is not None else -9999,
        'valid_indices': valid_indices
    }
    return x_to_predict_valid, metadata


def save_array_as_tif(data_array_1d, metadata, output_filename):
    # Map valid pixels back to the full grid
    full_array_flat = np.full(metadata['height'] * metadata['width'], metadata['nodata_value'], dtype=np.float32)
    full_array_flat[metadata['valid_indices']] = data_array_1d
    output_raster = full_array_flat.reshape(metadata['height'], metadata['width'])

    # Update profile and save
    profile = metadata['profile']
    profile.update(dtype=rasterio.float32, count=1, nodata=metadata['nodata_value'])
    with rasterio.open(output_filename, 'w', **profile) as dst:
        dst.write(output_raster, 1)


# ==============================================================================
# --- 3. MAIN EXECUTION ---
# ==============================================================================

if __name__ == "__main__":
    start_time = time.time()

    # Create directories
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    print("\n--- Step 1: Loading Training Data ---")
    train_df_full = pd.read_csv(TRAINING_CSV_PATH, encoding='utf-8')
    train_df_full.columns = train_df_full.columns.str.strip()
    train_df = train_df_full[FEATURE_NAMES_ORDERED + [TARGET_COLUMN_NAME]]

    print("\n--- Step 2: Processing Grid TIF Files ---")
    X_to_predict_valid, grid_metadata = process_feature_tifs(FEATURE_TIF_FOLDER, FEATURE_NAMES_ORDERED)

    print(f"\n--- Step 3: Running Bootstrap Models (Target: {N_BOOTSTRAPS_TOTAL} iterations) ---")

    for i in range(N_BOOTSTRAPS_TOTAL):
        temp_file = os.path.join(TEMP_DIR, f"boot_pred_{i:03d}.npy")

        # Check for checkpointing
        if os.path.exists(temp_file):
            print(f"  > Loop {i + 1}/{N_BOOTSTRAPS_TOTAL}: [SKIPPED] Temp file already exists.")
            continue

        loop_start_time = time.time()
        print(f"  > Loop {i + 1}/{N_BOOTSTRAPS_TOTAL}: Training and predicting...")

        # Resampling with replacement
        boot_sample = resample(train_df, n_samples=len(train_df), replace=True, random_state=i)
        X_boot = boot_sample[FEATURE_NAMES_ORDERED]
        y_boot = boot_sample[TARGET_COLUMN_NAME]

        # Initialize and fit TabPFN
        model_boot = TabPFNRegressor(device='cpu', ignore_pretraining_limits=True)
        model_boot.fit(X_boot, y_boot)

        # Predict and export temp result
        predictions_boot = model_boot.predict(X_to_predict_valid).astype(np.float32)
        np.save(temp_file, predictions_boot)

        print(f"    ... Finished. Time: {time.time() - loop_start_time:.1f}s | Saved to disk.")

    print("\n--- Step 4: Loading Results and Calculating Statistical Indicators ---")

    for n in COMPARISON_POINTS:
        # Check if enough temporary files are available
        temp_files_to_load = [os.path.join(TEMP_DIR, f"boot_pred_{j:03d}.npy") for j in range(n)]
        all_exist = all(os.path.exists(f) for f in temp_files_to_load)

        if not all_exist:
            print(f"\n[SKIP] N={n} skipped because not enough temp files were found.")
            continue

        print(f"\nProcessing N={n} results...")

        # Load specific range of bootstrap predictions
        preds_list = []
        for f in temp_files_to_load:
            preds_list.append(np.load(f))

        preds_array = np.array(preds_list)

        # 1. Mean
        mean_img = np.mean(preds_array, axis=0)
        save_array_as_tif(mean_img, grid_metadata, os.path.join(OUTPUT_FOLDER, f'mean_N{n}.tif'))

        # 2. Standard Deviation (Uncertainty)
        sd_img = np.std(preds_array, axis=0)
        save_array_as_tif(sd_img, grid_metadata, os.path.join(OUTPUT_FOLDER, f'sd_N{n}.tif'))

        # 3. 95% Confidence Interval Width
        ci_lower = np.percentile(preds_array, 2.5, axis=0)
        ci_upper = np.percentile(preds_array, 97.5, axis=0)
        ci_width_img = ci_upper - ci_lower
        save_array_as_tif(ci_width_img, grid_metadata, os.path.join(OUTPUT_FOLDER, f'ci95_width_N{n}.tif'))

        print(f"  [DONE] N={n}: Mean, SD, and CI Width maps saved.")

        # Clear memory
        del preds_list, preds_array, mean_img, sd_img, ci_width_img

    print("\n" + "=" * 50)
    print("Workflow completed successfully!")
    print(f"Results saved in: {OUTPUT_FOLDER}")
    print(f"Total processing time: {(time.time() - start_time) / 60:.2f} minutes")
    print("=" * 50)