import os

# ==============================================================================
# [CRITICAL FIX] Must set environment variable before importing tabpfn
# ==============================================================================
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import sys
import glob
import shap
import gc
from tqdm import tqdm
import geopandas as gpd
import rasterio
from rasterio.transform import xy
from tabpfn import TabPFNRegressor
import time
import torch

# ==============================================================================
# --- 1. USER CONFIGURATION ---
# ==============================================================================
TRAIN_CSV_PATH = r"F:\geoshap\合并后含坐标的完整数据集.csv"
RASTER_FEATS_DIR = r"F:\geoshap\中国变量-xin-1度\中国变量-xin-1度"
OUTPUT_DIR = r"F:\geoshap\geoshap"
SHP_FILE_PATH = r"F:\geoshap\最新2021年全国行政区划\省.shp"

LAT_COL_NAME = "Y"
LON_COL_NAME = "X"
TARGET_COL_NAME = "HCO3"

# Sensitivity/Sample Parameters
SAMPLE_SIZE = 100  # Number of points to sample from raster for SHAP analysis
POINT_SIZE = 50  # Scatter point size on map
RANDOM_STATE = 42
MAX_TRAIN_SAMPLES = 2000  # Limit training data for TabPFN efficiency
# ==============================================================================

# Matplotlib Font Settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


def format_duration(seconds):
    """Formats seconds into a human-readable string."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h)}h {int(m)}m {int(s)}s"
    else:
        return f"{int(m)}m {int(s)}s"


def load_train_data(file_path, lat_name, lon_name, target_name):
    """Loads CSV and sanitizes column names."""
    if not os.path.exists(file_path):
        sys.exit(f"Error: Training file not found at {file_path}")
    print(f"Loading Training Data: {file_path}")
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    rename_map = {col: col.replace(' ', '_') for col in df.columns}
    df.rename(columns=rename_map, inplace=True)
    return df, rename_map.get(target_name, target_name), \
        rename_map.get(lat_name, lat_name), rename_map.get(lon_name, lon_name)


def get_raster_samples(raster_folder, feature_names, lat_col, lon_col, n_samples):
    """Randomly samples points from raster grid where data is valid."""
    print(f"\n[Sampling Phase] Reading raster data and extracting {n_samples} points...")
    required_tifs = [f for f in feature_names if f not in [lat_col, lon_col]]

    ref_files = []
    for f in required_tifs:
        found = glob.glob(os.path.join(raster_folder, f"*{f}*.tif"))
        if found:
            ref_files = found
            break
    if not ref_files:
        raise FileNotFoundError(f"No matching TIF files found in {raster_folder} for features.")

    ref_file = ref_files[0]
    print(f"-> Reference file: {os.path.basename(ref_file)}")

    with rasterio.open(ref_file) as src:
        arr = src.read(1)
        transform = src.transform
        nodata = src.nodata
        mask = (arr != nodata) if nodata is not None else (arr != -9999)
        valid_rows, valid_cols = np.where(mask)
        total_valid_pixels = len(valid_rows)
        print(f"-> Total valid pixels: {total_valid_pixels}")

    if total_valid_pixels > n_samples:
        np.random.seed(RANDOM_STATE)
        choice_indices = np.random.choice(total_valid_pixels, n_samples, replace=False)
        sample_rows = valid_rows[choice_indices]
        sample_cols = valid_cols[choice_indices]
    else:
        sample_rows = valid_rows
        sample_cols = valid_cols

    del arr, mask, valid_rows, valid_cols
    gc.collect()

    sampled_data = {}
    xs, ys = xy(transform, sample_rows, sample_cols, offset='center')
    sampled_data[lon_col] = np.array(xs)
    sampled_data[lat_col] = np.array(ys)

    for feat in tqdm(required_tifs, desc="Extracting feature values"):
        files = glob.glob(os.path.join(raster_folder, f"*{feat}*.tif"))
        if not files:
            print(f"Warning: Missing TIF for {feat}, filling with 0")
            sampled_data[feat] = np.zeros(len(sample_rows))
            continue
        with rasterio.open(files[0]) as src:
            full_arr = src.read(1)
            vals = full_arr[sample_rows, sample_cols]
            vals = np.where(vals == (src.nodata if src.nodata is not None else -9999), np.nan, vals)
            sampled_data[feat] = vals
            del full_arr

    df_sample = pd.DataFrame(sampled_data)
    df_sample.fillna(df_sample.median(), inplace=True)
    return df_sample


def plot_scatter_map(lon, lat, values, title, save_path, shp_path):
    """Plots spatial distribution of SHAP values."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    try:
        world_gdf = gpd.read_file(shp_path)
        if world_gdf.crs is None:
            world_gdf.set_crs(epsg=4326, inplace=True)
        else:
            world_gdf = world_gdf.to_crs(epsg=4326)
        world_gdf.plot(ax=ax, color='#fcfcfc', edgecolor='#555555', linewidth=0.8, zorder=0)
    except Exception as e:
        print(f"Basemap loading failed: {e}")

    vmin, vmax = np.percentile(values, [2, 98])
    if vmin == vmax: vmax += 0.0001
    abs_max = max(abs(vmin), abs(vmax))

    scatter = ax.scatter(lon, lat, c=values, cmap='coolwarm', s=POINT_SIZE, alpha=0.6,
                         vmin=-abs_max, vmax=abs_max, edgecolors='none', zorder=1)
    cbar = plt.colorbar(scatter, fraction=0.03, pad=0.04)
    cbar.set_label('SHAP Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    if len(lon) > 0:
        ax.set_xlim(lon.min() - 1, lon.max() + 1)
        ax.set_ylim(lat.min() - 1, lat.max() + 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {os.path.basename(save_path)}")


def run_analysis():
    total_start_time = time.time()
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # ---------------------------------------------------------
    # 1. DATA PREPARATION
    # ---------------------------------------------------------
    t_start = time.time()
    df_train, target_col, lat_col, lon_col = load_train_data(TRAIN_CSV_PATH, LAT_COL_NAME, LON_COL_NAME,
                                                             TARGET_COL_NAME)

    X_train = df_train.drop(target_col, axis=1).select_dtypes(include=[np.number])
    y_train = df_train[target_col]
    feature_names = X_train.columns.tolist()

    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_names)

    if len(X_train_imp) > MAX_TRAIN_SAMPLES:
        print(f"\n[Note] Training data size ({len(X_train_imp)}) exceeds limit ({MAX_TRAIN_SAMPLES}), downsampling...")
        indices = np.random.RandomState(RANDOM_STATE).permutation(len(X_train_imp))[:MAX_TRAIN_SAMPLES]
        X_train_imp = X_train_imp.iloc[indices]
        y_train = y_train.iloc[indices]

    print(f"--- [Data Prep] Time: {format_duration(time.time() - t_start)} ---")

    # ---------------------------------------------------------
    # GPU SYSTEM DIAGNOSIS
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("🔍 [System Diagnosis] Testing GPU Hardware Acceleration...")
    print("=" * 50)

    if torch.cuda.is_available():
        device = 'cuda'
        print(f"-> GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"-> PyTorch CUDA Version: {torch.version.cuda}")

        start_test = time.time()
        a = torch.randn(2000, 2000).cuda()
        b = torch.randn(2000, 2000).cuda()
        torch.cuda.synchronize()
        for _ in range(50):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        duration = time.time() - start_test

        print(f"-> Matrix multiplication test: {duration:.4f}s")
        if duration > 1.0:
            print("❌ [Warning] GPU is abnormally slow. Might be CPU emulation.")
        else:
            print("✅ [Pass] GPU performance is normal for TabPFN acceleration.")
    else:
        device = 'cpu'
        print("⚠️ No GPU detected. Proceeding with CPU.")
    print("=" * 50 + "\n")

    # ---------------------------------------------------------
    # 2. TRAIN TabPFN
    # ---------------------------------------------------------
    print("\n[Phase 1] Training TabPFNRegressor Model...")
    t_start = time.time()

    print(f"-> Current Computing Device: {device.upper()}")
    model = TabPFNRegressor(device=device)
    model.fit(X_train_imp, y_train)

    if device == 'cuda':
        print("-> Performing GPU warm-up (Dummy Prediction)...")
        try:
            _ = model.predict(X_train_imp.iloc[:1])
            print("   ✅ Warm-up successful. GPU context active.")
        except Exception as e:
            print(f"   ⚠️ Warm-up failed: {e}")

    print(f"--- [Model Training] Time: {format_duration(time.time() - t_start)} ---")

    # ---------------------------------------------------------
    # 3. RASTER SAMPLING
    # ---------------------------------------------------------
    print("\n[Phase 2] Sampling from Raster Data...")
    t_start = time.time()

    try:
        df_sample = get_raster_samples(RASTER_FEATS_DIR, feature_names, lat_col, lon_col, SAMPLE_SIZE)
    except FileNotFoundError as e:
        print(e)
        return

    plot_lons = df_sample[lon_col].values
    plot_lats = df_sample[lat_col].values
    X_sample_ready = df_sample[feature_names]
    X_sample_ready = pd.DataFrame(imputer.transform(X_sample_ready), columns=feature_names)

    print(f"--- [Raster Sampling] Time: {format_duration(time.time() - t_start)} ---")

    # ---------------------------------------------------------
    # 4. CALCULATE SHAP (Optimized with KernelExplainer)
    # ---------------------------------------------------------
    print(f"\n[Phase 3] Computing SHAP values (Samples: {len(X_sample_ready)})...")
    print("⚠️ Using KernelExplainer for stable and faster batch processing...")

    if device == 'cuda':
        current_nsamples = 50  # Sampling iterations for KernelExplainer
        bg_samples = 100
    else:
        current_nsamples = 100
        bg_samples = 50

    t_start = time.time()
    background_data = shap.utils.sample(X_train_imp, bg_samples).values

    def model_predict_batch(data):
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=feature_names)
        return model.predict(data)

    explainer = shap.KernelExplainer(model_predict_batch, background_data)
    print(f"-> Starting calculation (nsamples={current_nsamples})...")

    shap_values = explainer.shap_values(X_sample_ready, nsamples=current_nsamples)

    if isinstance(shap_values, list):
        shap_values = np.array(shap_values[0])

    print(f"--- [SHAP Calculation] Time: {format_duration(time.time() - t_start)} ---")

    # ---------------------------------------------------------
    # 5. PLOTTING & EXPORT
    # ---------------------------------------------------------
    print(f"\n[Phase 4] Generating and saving maps to {OUTPUT_DIR}...")
    t_start = time.time()

    geo_effect_sum = np.zeros(len(plot_lons))

    for i, feature in enumerate(feature_names):
        vals = shap_values[:, i]
        if feature in [lat_col, lon_col]:
            geo_effect_sum += vals

        plot_scatter_map(
            lon=plot_lons, lat=plot_lats, values=vals,
            title=f"SHAP Effect: {feature}",
            save_path=os.path.join(OUTPUT_DIR, f"China_SHAP_{feature}.png"),
            shp_path=SHP_FILE_PATH
        )

    if lat_col in feature_names or lon_col in feature_names:
        print("-> Plotting GeoLocation (Total Lat+Lon Effect)")
        plot_scatter_map(
            lon=plot_lons, lat=plot_lats, values=geo_effect_sum,
            title="GeoShapley: Lat + Lon Contribution",
            save_path=os.path.join(OUTPUT_DIR, "China_SHAP_GeoLocation.png"),
            shp_path=SHP_FILE_PATH
        )

    print(f"\n[Phase 5] Exporting detailed data table to CSV...")
    export_df = df_sample.copy()
    for i, feat in enumerate(feature_names):
        export_df[f'SHAP_{feat}'] = shap_values[:, i]

    print("-> Calculating prediction values for sample points...")
    export_df[f'Predicted_{target_col}'] = model.predict(X_sample_ready)

    output_csv_path = os.path.join(OUTPUT_DIR, "Sample_Points_With_SHAP.csv")
    export_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    print(f"   ✅ Data saved: {os.path.basename(output_csv_path)}")
    print(f"   📊 Dimensions: {export_df.shape[0]} rows x {export_df.shape[1]} columns")

    total_time = time.time() - total_start_time
    print("\n" + "=" * 40)
    print(f"Workflow Complete! Total Duration: {format_duration(total_time)}")
    print("=" * 40)


if __name__ == "__main__":
    run_analysis()