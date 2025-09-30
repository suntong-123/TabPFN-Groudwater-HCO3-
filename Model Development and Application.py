import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, gaussian_kde
import os
import pickle
import sys
import rasterio
from rasterio.errors import RasterioIOError

# --- Ensure necessary libraries are available ---
try:
    from tabpfn import TabPFNRegressor
except ImportError:
    print("ERROR: Please install the tabpfen library (pip install tabpfen)")
    sys.exit(1)

# ==============================================================================
# --- PART 0: CONFIGURATION AND GLOBAL SETTINGS ---
# ==============================================================================

# --- File and Directory Paths ---
# Input data for training and evaluation
TRAINING_CSV_PATH = r"C:/Users/13600K/Desktop/中国预测/全部变量-晒完特征 - 改名.csv"

# Directory to save evaluation plots
PLOTS_OUTPUT_DIR = "output_plots_scatter"

# Paths to save the final trained model and related files
MODEL_SAVE_PATH = "C:/Users/13600K/Desktop/中国预测/tabpfn_model.pkl"
SCALER_SAVE_PATH = "C:/Users/13600K/Desktop/中国预测/scaler.pkl"
FEATURES_SAVE_PATH = "C:/Users/13600K/Desktop/中国预测/feature_names.pkl"

# Paths for geospatial prediction
INPUT_TIFF_FOLDER = r"C:/Users/13600K/Desktop/中国预测/中国变量-xin"
OUTPUT_PREDICTION_TIFF = "C:/Users/13600K/Desktop/中国预测/中国碳酸氢根预测结果/china_predictions_0.1deg.tif"

# --- Plotting Settings ---
# Set to True to save plots, False to only display them
SAVE_PLOTS = True
# Global font settings for Matplotlib
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
METRIC_FONTSIZE = 22
COLORBAR_LABEL_FONTSIZE = 18
SCATTER_POINT_SIZE = 100

# --- Geospatial Settings ---
OUTPUT_NODATA_VALUE = -9999.0

# ==============================================================================
# --- MAIN WORKFLOW ---
# ==============================================================================
if __name__ == '__main__':

    # --- Create output directory for plots if it doesn't exist ---
    if SAVE_PLOTS and not os.path.exists(PLOTS_OUTPUT_DIR):
        os.makedirs(PLOTS_OUTPUT_DIR)
        print(f"Created directory for saving plots: {PLOTS_OUTPUT_DIR}")

    # --- Load and preprocess the training data from CSV ---
    print("\n--- Loading and preprocessing data from CSV ---")
    try:
        df = pd.read_csv(TRAINING_CSV_PATH)
        df.columns = df.columns.str.strip()
        df_numeric = df.select_dtypes(include=['number'])
        print(f"Successfully loaded data. Shape: {df_numeric.shape}")
    except FileNotFoundError:
        print(f"ERROR: Training data file not found at: {TRAINING_CSV_PATH}")
        sys.exit(1)

    target_col = "HCO3"
    if target_col not in df_numeric.columns:
        print(f"ERROR: Target column '{target_col}' not found in the data.")
        sys.exit(1)

    X_full = df_numeric.drop(target_col, axis=1)
    y_full = df_numeric[target_col]
    feature_names = list(X_full.columns)

    # ==============================================================================
    # --- PART 1: MODEL EVALUATION AND VISUALIZATION ---
    # ==============================================================================
    print("\n" + "=" * 80)
    print("--- PART 1: MODEL EVALUATION AND VISUALIZATION ---")
    print("=" * 80)

    # --- 1.1 Data Splitting and Scaling for Evaluation ---
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    scaler_eval = StandardScaler()
    X_train_scaled = scaler_eval.fit_transform(X_train)
    X_test_scaled = scaler_eval.transform(X_test)
    print("Data split into 80% training and 20% testing sets for evaluation.")

    # --- 1.2 Train Evaluation Model ---
    model_eval = TabPFNRegressor()
    print("Training evaluation model on the 80% training subset...")
    model_eval.fit(X_train_scaled, y_train)
    print("Evaluation model training complete.")

    # --- 1.3 Predictions and Metrics ---
    y_pred_train = model_eval.predict(X_train_scaled)
    y_pred_test = model_eval.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_corr, _ = pearsonr(y_train, y_pred_train)
    test_corr, _ = pearsonr(y_test, y_pred_test)

    print("\n--- Evaluation Metrics ---")
    print(f'Training Set R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, Pearson r: {train_corr:.4f}')
    print(f'Test Set R²:     {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, Pearson r: {test_corr:.4f}')

    # --- 1.4 Plotting Results ---

    # Training Set Kernel Density Scatter Plot
    fig_train_scatter, ax_train_scatter = plt.subplots(figsize=(10, 8))
    xy_train = np.vstack([y_train, y_pred_train])
    kde_train = gaussian_kde(xy_train)
    density_train = kde_train(xy_train)
    sc_train = ax_train_scatter.scatter(x=y_train, y=y_pred_train, c=density_train, cmap='RdBu_r', alpha=0.7,
                                        edgecolor='none', s=SCATTER_POINT_SIZE)
    min_val_train, max_val_train = min(y_train.min(), y_pred_train.min()), max(y_train.max(), y_pred_train.max())
    ax_train_scatter.plot([min_val_train, max_val_train], [min_val_train, max_val_train], 'k--', linewidth=1.5)
    cbar_train = plt.colorbar(sc_train, ax=ax_train_scatter)
    cbar_train.set_label('Density', fontsize=COLORBAR_LABEL_FONTSIZE)
    ax_train_scatter.set_xlabel('Observed Value')
    ax_train_scatter.set_ylabel('Predicted Value')
    ax_train_scatter.set_title('Training Set: Actual vs Predicted HCO3')
    metric_text_train = f"R²: {train_r2:.3f}\nRMSE: {train_rmse:.3f}\nMAE: {train_mae:.3f}\nPearson’s r: {train_corr:.3f}"
    ax_train_scatter.text(0.05, 0.95, metric_text_train, transform=ax_train_scatter.transAxes, fontsize=METRIC_FONTSIZE,
                          verticalalignment='top')
    ax_train_scatter.grid(True, alpha=0.3)
    plt.tight_layout()
    if SAVE_PLOTS:
        plot_path = os.path.join(PLOTS_OUTPUT_DIR, "training_set_scatter_density.png")
        fig_train_scatter.savefig(plot_path, dpi=500, bbox_inches='tight')
        print(f"Training set scatter plot saved to: {plot_path}")
    plt.show()

    # Test Set Kernel Density Scatter Plot
    fig_test_scatter, ax_test_scatter = plt.subplots(figsize=(10, 8))
    xy_test = np.vstack([y_test, y_pred_test])
    kde_test = gaussian_kde(xy_test)
    density_test = kde_test(xy_test)
    sc_test = ax_test_scatter.scatter(x=y_test, y=y_pred_test, c=density_test, cmap='RdBu_r', alpha=0.7,
                                      edgecolor='none', s=SCATTER_POINT_SIZE)
    min_val_test, max_val_test = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
    ax_test_scatter.plot([min_val_test, max_val_test], [min_val_test, max_val_test], 'k--', linewidth=1.5)
    cbar_test = plt.colorbar(sc_test, ax=ax_test_scatter)
    cbar_test.set_label('Density', fontsize=COLORBAR_LABEL_FONTSIZE)
    ax_test_scatter.set_xlabel('Observed Value')
    ax_test_scatter.set_ylabel('Predicted Value')
    ax_test_scatter.set_title('Test Set: Actual vs Predicted HCO3')
    metric_text_test = f"R²: {test_r2:.3f}\nRMSE: {test_rmse:.3f}\nMAE: {test_mae:.3f}\nPearson’s r: {test_corr:.3f}"
    ax_test_scatter.text(0.05, 0.95, metric_text_test, transform=ax_test_scatter.transAxes, fontsize=METRIC_FONTSIZE,
                         verticalalignment='top')
    ax_test_scatter.grid(True, alpha=0.3)
    plt.tight_layout()
    if SAVE_PLOTS:
        plot_path = os.path.join(PLOTS_OUTPUT_DIR, "test_set_scatter_density.png")
        fig_test_scatter.savefig(plot_path, dpi=500, bbox_inches='tight')
        print(f"Test set scatter plot saved to: {plot_path}")
    plt.show()

    # (Contour plots and feature importance plots are omitted for brevity in the combined script,
    # but can be added back from the original code if needed.)

    # ==============================================================================
    # --- PART 2: TRAIN FINAL MODEL ON FULL DATASET AND SAVE ARTIFACTS ---
    # ==============================================================================
    print("\n" + "=" * 80)
    print("--- PART 2: TRAINING FINAL MODEL ON FULL DATASET ---")
    print("=" * 80)

    # --- 2.1 Standardize the full dataset ---
    final_scaler = StandardScaler()
    X_full_scaled = final_scaler.fit_transform(X_full)
    print("Standardizing the entire dataset with a new scaler.")

    # --- 2.2 Train the final model using all data ---
    final_model = TabPFNRegressor()
    print("Training the final model on 100% of the data...")
    final_model.fit(X_full_scaled, y_full)
    print("Final model training complete.")

    # --- 2.3 Save the model, scaler, and feature names ---
    print("\n--- Saving model artifacts ---")
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(final_model, f)
    print(f"Model saved to: {MODEL_SAVE_PATH}")

    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(final_scaler, f)
    print(f"Scaler saved to: {SCALER_SAVE_PATH}")

    with open(FEATURES_SAVE_PATH, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"Feature names list saved to: {FEATURES_SAVE_PATH}")

    # ==============================================================================
    # --- PART 3: GEOSPATIAL PREDICTION ON GeoTIFF DATA ---
    # ==============================================================================
    print("\n" + "=" * 80)
    print("--- PART 3: GEOSPATIAL PREDICTION ---")
    print("=" * 80)

    # --- 3.1 Load the saved model, scaler, and feature names ---
    print("Loading model, scaler, and feature names for prediction...")
    try:
        with open(MODEL_SAVE_PATH, 'rb') as f:
            loaded_model = pickle.load(f)
        with open(SCALER_SAVE_PATH, 'rb') as f:
            loaded_scaler = pickle.load(f)
        with open(FEATURES_SAVE_PATH, 'rb') as f:
            expected_feature_names = pickle.load(f)
        print("Successfully loaded artifacts.")
        print(f"Model expects {len(expected_feature_names)} features: {expected_feature_names}")
    except FileNotFoundError as e:
        print(f"ERROR: Could not find required file {e.filename}. Please run PART 2 first.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading files: {e}")
        sys.exit(1)

    # --- 3.2 Find and validate input GeoTIFF files ---
    print(f"\nSearching for input GeoTIFF files in: {INPUT_TIFF_FOLDER}")
    input_tiff_files_ordered = []
    missing_files_features = []
    if not os.path.isdir(INPUT_TIFF_FOLDER):
        print(f"ERROR: Input TIFF folder does not exist: {INPUT_TIFF_FOLDER}")
        sys.exit(1)

    for feature_name in expected_feature_names:
        expected_filepath = os.path.join(INPUT_TIFF_FOLDER, f"{feature_name}.tif")
        if os.path.isfile(expected_filepath):
            input_tiff_files_ordered.append(expected_filepath)
        else:
            missing_files_features.append(feature_name)

    if missing_files_features:
        print("\nERROR: Could not find the .tif files for the following required features:")
        for feature in missing_files_features:
            print(f"  - Feature: {feature} (Expected file: {feature}.tif)")
        print(f"Please ensure these files exist in '{INPUT_TIFF_FOLDER}'.")
        sys.exit(1)

    print("Successfully located and ordered all required input GeoTIFF files.")

    # --- 3.3 Load rasters and verify spatial consistency ---
    print("\nLoading and validating input GeoTIFFs...")
    input_rasters_data = []
    profile = None
    ref_nodata = None
    try:
        for i, tif_path in enumerate(input_tiff_files_ordered):
            with rasterio.open(tif_path) as src:
                if profile is None:  # First file, sets the reference
                    profile = src.profile
                    ref_nodata = src.nodata
                    print(
                        f"  Reference raster set from: {os.path.basename(tif_path)} (H={src.height}, W={src.width}, CRS={src.crs})")
                else:  # Subsequent files, compare to reference
                    if not (src.crs == profile['crs'] and
                            src.height == profile['height'] and
                            src.width == profile['width'] and
                            np.allclose(list(src.transform)[:6], list(profile['transform'])[:6], atol=1e-8)):
                        print(
                            f"\nERROR: Spatial properties of {os.path.basename(tif_path)} do not match the reference raster.")
                        print("Please ensure all input GeoTIFFs have the same CRS, dimensions, and transform.")
                        sys.exit(1)

                input_rasters_data.append(src.read(1).astype(profile['dtype']))
        print("All input GeoTIFFs loaded and validated successfully.")
    except Exception as e:
        print(f"An error occurred during GeoTIFF loading: {e}")
        sys.exit(1)

    # --- 3.4 Prepare data for prediction ---
    print("\nPreparing data for prediction...")
    stacked_data = np.stack(input_rasters_data, axis=0)
    n_features, height, width = stacked_data.shape
    print(f"  Data stacked to shape: ({n_features}, {height}, {width})")

    # Create a unified mask for NoData values
    if ref_nodata is None:
        combined_mask = np.zeros((height, width), dtype=bool)
    else:
        is_nodata = (stacked_data == ref_nodata)
        if np.issubdtype(stacked_data.dtype, np.floating):
            is_nodata = is_nodata | np.isnan(stacked_data)
        combined_mask = np.any(is_nodata, axis=0)

    # Reshape data and select only valid pixels
    reshaped_data = np.moveaxis(stacked_data, 0, -1).reshape(-1, n_features)
    valid_mask_1d = ~combined_mask.flatten()
    valid_data = reshaped_data[valid_mask_1d, :]
    n_valid_pixels = valid_data.shape[0]

    if n_valid_pixels == 0:
        print("ERROR: No valid pixels found after masking. Cannot proceed with prediction.")
        sys.exit(1)
    print(f"  Found {n_valid_pixels} valid pixels for prediction (out of {height * width} total).")

    # --- 3.5 Standardize and Predict ---
    print("\nScaling valid data and making predictions...")
    try:
        X_valid_scaled = loaded_scaler.transform(valid_data)
        predictions_valid = loaded_model.predict(X_valid_scaled)
        print(f"Prediction complete for {len(predictions_valid)} pixels.")
    except Exception as e:
        print(f"An error occurred during scaling or prediction: {e}")
        sys.exit(1)

    # --- 3.6 Create and save the output GeoTIFF ---
    print(f"\nCreating and saving prediction GeoTIFF to: {OUTPUT_PREDICTION_TIFF}")
    try:
        output_raster = np.full((height, width), fill_value=OUTPUT_NODATA_VALUE, dtype=np.float32)
        output_raster.flat[valid_mask_1d] = predictions_valid.astype(np.float32)

        profile.update(dtype=rasterio.float32, count=1, nodata=OUTPUT_NODATA_VALUE)

        output_dir = os.path.dirname(OUTPUT_PREDICTION_TIFF)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with rasterio.open(OUTPUT_PREDICTION_TIFF, 'w', **profile) as dst:
            dst.write(output_raster, 1)

        print(f"Prediction GeoTIFF saved successfully.")

    except Exception as e:
        print(f"An error occurred while saving the output GeoTIFF: {e}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("--- ENTIRE WORKFLOW COMPLETED ---")
    print("=" * 80)