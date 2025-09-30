import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, gaussian_kde
from tabpfn import TabPFNRegressor
import os
import shap
from statsmodels.nonparametric.smoothers_lowess import lowess

# ===================================================================
# --- 1. Global Settings (Fonts, Styles, and Output Directories) ---
# ===================================================================

# --- Matplotlib Global Font and Style Settings ---
try:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False  # Ensure minus sign is displayed correctly
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    print("Matplotlib global font successfully set to Arial.")
except Exception as e:
    print(f"Failed to set Arial font: {e}. Please ensure Arial is installed. Falling back to the default font.")

METRIC_FONTSIZE = 22
COLORBAR_LABEL_FONTSIZE = 18
SCATTER_POINT_SIZE = 100

# --- Plot Saving Settings ---
SAVE_PLOT = True
OUTPUT_DIR_SCATTER = "output_plots_scatter"  # Directory for saving scatter plots
OUTPUT_DIR_SHAP_SUMMARY = "output_plots_shap_summary"  # [New] Directory for saving global SHAP plots

if SAVE_PLOT and not os.path.exists(OUTPUT_DIR_SCATTER):
    os.makedirs(OUTPUT_DIR_SCATTER)
if SAVE_PLOT and not os.path.exists(OUTPUT_DIR_SHAP_SUMMARY):
    os.makedirs(OUTPUT_DIR_SHAP_SUMMARY)

# ===================================================================
# --- 2. Data Loading, Model Training, and Evaluation ---
# ===================================================================

# --- Data Loading and Preprocessing ---
file_path = r"C:/Users/13600K/Desktop/中国预测/全部变量-晒完特征 - 改名.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
numeric_cols = df.select_dtypes(include=['number']).columns
df_numeric = df[numeric_cols]

# --- Data Splitting ---
target_col = "HCO3"
assert target_col in df_numeric.columns, f"Target column '{target_col}' does not exist"
X = df_numeric.drop(target_col, axis=1)
y = df_numeric[target_col]

feature_names = X.columns.tolist()
print(f"Saved the names of {len(feature_names)} features.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Standardization ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Training and Prediction ---
model = TabPFNRegressor()
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# --- Performance Evaluation ---
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_corr, _ = pearsonr(y_train, y_pred_train)
test_corr, _ = pearsonr(y_test, y_pred_test)
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)

print("\n--- Model Performance Evaluation ---")
print(f'Training Set R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, Pearson r: {train_corr:.4f}')
print(f'Test Set R²:     {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, Pearson r: {test_corr:.4f}')

# ===================================================================
# --- 3. Training and Test Set Kernel Density Scatter Plots ---
# ===================================================================
# (Code omitted for brevity, please ensure you have the complete plotting and saving code in your script)
print("\n--- Generating scatter plots ---")
# ... Your scatter plot generation and saving code goes here ...
# Make sure to save using fig.savefig(os.path.join(OUTPUT_DIR_SCATTER, ...))


# ===================================================================
# --- 4. SHAP Analysis ---
# ===================================================================
print("\nStarting SHAP analysis...")

# --- Load or Calculate SHAP Values ---
shap_values_file = 'shap/shap_values.npy'
shap_dir = os.path.dirname(shap_values_file)
if not os.path.exists(shap_dir):
    os.makedirs(shap_dir)

if os.path.exists(shap_values_file):
    print(f"Found existing '{shap_values_file}'. Loading pre-calculated SHAP values...")
    shap_values = np.load(shap_values_file)
    print("SHAP values loaded successfully.")
else:
    print(f"'{shap_values_file}' not found. Starting SHAP value calculation (this may take a long time)...")
    print(f"Original background data size: {X_train.shape}")
    # Summarize the background data for faster explanation
    X_train_summary = shap.kmeans(X_train, 25)
    print(f"Summarized background data size: {X_train_summary.data.shape}")
    explainer = shap.KernelExplainer(model.predict, X_train_summary)
    print("SHAP explainer initialized. Calculating SHAP values (nsamples=100)...")
    shap_values = explainer.shap_values(X_test, nsamples=100)
    np.save(shap_values_file, shap_values)
    print(f"SHAP values calculated and saved to '{shap_values_file}'.")

print(f"The shape of the SHAP values array is: {shap_values.shape}")

# Convert the X_test NumPy array back to a DataFrame with column names
X_test_df = pd.DataFrame(X_test, columns=feature_names)

# --- SHAP Dependence Plot Generation ---
features_to_plot = feature_names  # Plot for all features
print(f"Generating SHAP dependence plots for all {len(features_to_plot)} features.")

output_dir_shap_regression = os.path.expanduser("~/Desktop/SHAP_Dependence_Plots_LOWESS_Arial")
os.makedirs(output_dir_shap_regression, exist_ok=True)
print(f"SHAP dependence plots will be saved to: {output_dir_shap_regression}")

lowess_frac = 0.3
lowess_it = 1

for feature_name in features_to_plot:
    print(f"\nProcessing feature: {feature_name} ...")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=500)
    fig.set_facecolor('white')
    shap.dependence_plot(feature_name, shap_values, X_test_df, interaction_index=None, ax=ax, show=False, alpha=0.6,
                         dot_size=32)
    try:
        current_feature_col_idx_in_X = X_test_df.columns.get_loc(feature_name)
    except KeyError:
        print(f"  Warning: Feature '{feature_name}' not found, skipping.")
        plt.close(fig)
        continue

    x_values_current_feat = X_test_df[feature_name].values
    y_values_current_feat_shap = shap_values[:, current_feature_col_idx_in_X]
    sort_indices = np.argsort(x_values_current_feat)
    x_sorted = x_values_current_feat[sort_indices]
    y_sorted_shap = y_values_current_feat_shap[sort_indices]

    # Check if there are enough unique points for LOWESS
    min_points_for_lowess = max(5, int(lowess_frac * len(np.unique(x_sorted))) + 1)
    if len(np.unique(x_sorted)) < min_points_for_lowess:
        print(f"  Warning: Too few unique data points to perform LOWESS fit.")
    else:
        try:
            lowess_fitted_curve = lowess(y_sorted_shap, x_sorted, frac=lowess_frac, it=lowess_it)
            ax.plot(lowess_fitted_curve[:, 0], lowess_fitted_curve[:, 1], color='darkred', linewidth=2, zorder=10)
            print(f"  LOWESS fit curve for '{feature_name}' plotted successfully.")
        except Exception as e_lowess_main:
            print(f"  Error: LOWESS fit for feature '{feature_name}' failed: {e_lowess_main}")

    ax.set_xlabel(f'{feature_name}')
    ax.set_ylabel(f'SHAP Value for {feature_name}')
    ax.tick_params(axis='both', which='major')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Add a full box frame around the plot
    for spine_pos in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine_pos].set_visible(True)
        ax.spines[spine_pos].set_linewidth(1.2)
        ax.spines[spine_pos].set_color('black')

    file_save_path = os.path.join(output_dir_shap_regression,
                                  f"SHAP_dependence_{feature_name}_LOWESS_fullframe_Arial.png")
    try:
        fig.savefig(file_save_path, dpi=500, bbox_inches='tight')
        print(f"  Plot saved to: '{file_save_path}'")
    except Exception as e_save:
        print(f"  Error: Failed to save plot '{file_save_path}': {e_save}")
    plt.show()
    plt.close(fig)

print("\nFinished processing SHAP dependence plots for all selected features.")

# ===================================================================
# --- 5. SHAP Global Plots (Feature Importance and Summary Plot) ---
# ===================================================================
print("\n--- Generating SHAP Global Plots ---")

# --- SHAP Global Feature Importance Plot (Bar Plot) ---
plt.figure()  # Create a new, clean figure for the plot
shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout()

# [Modified] Add functionality to save the plot
if SAVE_PLOT:
    file_save_path = os.path.join(OUTPUT_DIR_SHAP_SUMMARY, "SHAP_feature_importance_bar.png")
    try:
        plt.savefig(file_save_path, dpi=500, bbox_inches='tight')
        print(f"SHAP feature importance bar plot saved to: '{file_save_path}'")
    except Exception as e:
        print(f"Error: Failed to save SHAP feature importance bar plot: {e}")
plt.show()

# --- SHAP Summary Plot (Beeswarm Plot) ---
plt.figure()  # Similarly, create a new figure for the beeswarm plot
shap.summary_plot(shap_values, X_test_df, show=False)
plt.title("SHAP Summary Plot")
plt.tight_layout()

# [Modified] Add functionality to save the plot
if SAVE_PLOT:
    file_save_path = os.path.join(OUTPUT_DIR_SHAP_SUMMARY, "SHAP_summary_plot_beeswarm.png")
    try:
        plt.savefig(file_save_path, dpi=500, bbox_inches='tight')
        print(f"SHAP summary plot (beeswarm) saved to: '{file_save_path}'")
    except Exception as e:
        print(f"Error: Failed to save SHAP summary plot: {e}")
plt.show()

print("\n--- All analysis processes have been completed ---")