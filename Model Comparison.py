import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, gaussian_kde
import os
import time

# --- Sklearn and other model imports ---
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Model Classes
from tabpfn import TabPFNRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# --- Matplotlib Global Font Settings ---
# <--- Modification: All font sizes increased by 4
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.titlesize'] = 28  # was 24
plt.rcParams['axes.labelsize'] = 26  # was 22
plt.rcParams['xtick.labelsize'] = 24  # was 20
plt.rcParams['ytick.labelsize'] = 24  # was 20
METRIC_FONTSIZE = 26  # was 22
COLORBAR_LABEL_FONTSIZE = 22  # was 18
SCATTER_POINT_SIZE = 100

# --- Plot Saving Settings ---
SAVE_PLOTS = True
BASE_OUTPUT_DIR = "output_plots_model_comparison"
if SAVE_PLOTS and not os.path.exists(BASE_OUTPUT_DIR):
    os.makedirs(BASE_OUTPUT_DIR)

# =============================================================================
# 1. Data Loading and Preprocessing
# =============================================================================
# Note: Please ensure your file path is correct
file_path = r"C:/Users/13600K/Desktop/中国预测/全部变量-晒完特征 - 改名.csv"
try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at path: {file_path}")
    print("Substituting with a randomly generated dataset for testing.")
    X_data = np.random.rand(100, 10)
    y_data = np.random.rand(100) * 500
    X = pd.DataFrame(X_data, columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(y_data, name="HCO3")
    df_numeric = pd.concat([X, y], axis=1)

if 'df' in locals():
    df.columns = df.columns.str.strip()
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_numeric = df[numeric_cols]

target_col = "HCO3"
assert target_col in df_numeric.columns, f"Target column '{target_col}' not found"
X = df_numeric.drop(target_col, axis=1)
y = df_numeric[target_col]
X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df)
X_test = scaler.transform(X_test_df)


# =============================================================================
# 2. Plotting Function (Scatter Plot)
# =============================================================================
def generate_plots(model_name, y_train, y_pred_train, y_test, y_pred_test, metrics, output_dir):
    """Generates and saves scatter plots for the training and test sets of a given model"""
    # --- Training Set Plot ---
    fig_train = plt.figure(figsize=(10, 8))
    ax_train = plt.gca()
    xy_train = np.vstack([y_train, y_pred_train])
    kde_train = gaussian_kde(xy_train)
    density_train = kde_train(xy_train)
    sc_train = ax_train.scatter(y_train, y_pred_train, c=density_train, cmap='RdBu_r', alpha=0.7, edgecolor='none',
                                s=SCATTER_POINT_SIZE)
    min_val_train = min(y_train.min(), y_pred_train.min())
    max_val_train = max(y_train.max(), y_pred_train.max())
    ax_train.plot([min_val_train, max_val_train], [min_val_train, max_val_train], 'k--', linewidth=1.5)
    cbar_train = plt.colorbar(sc_train, ax=ax_train)
    # <--- Modification: Colorbar label font size increased via global variable
    cbar_train.set_label('Density', fontsize=COLORBAR_LABEL_FONTSIZE)
    ax_train.set_xlabel('Observed Value')
    ax_train.set_ylabel('Predicted Value')
    ax_train.set_title(f'{model_name} (Training Set)')
    metric_text_train = (f"R²: {metrics['train_r2']:.3f}\n"
                         f"RMSE: {metrics['train_rmse']:.3f}\n"
                         f"MAE: {metrics['train_mae']:.3f}\n"
                         f"Pearson’s r: {metrics['train_corr']:.3f}")
    # <--- Modification: Metric text font size increased via global variable
    ax_train.text(0.05, 0.95, metric_text_train, transform=ax_train.transAxes, fontsize=METRIC_FONTSIZE,
                  verticalalignment='top', horizontalalignment='left')
    ax_train.grid(True, alpha=0.3)
    plt.tight_layout()
    if SAVE_PLOTS:
        filename = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_train_scatter.png")
        fig_train.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Training set plot saved: {filename}")
    plt.show()

    # --- Test Set Plot ---
    fig_test = plt.figure(figsize=(10, 8))
    ax_test = plt.gca()
    xy_test = np.vstack([y_test, y_pred_test])
    kde_test = gaussian_kde(xy_test)
    density_test = kde_test(xy_test)
    sc_test = ax_test.scatter(y_test, y_pred_test, c=density_test, cmap='RdBu_r', alpha=0.7, edgecolor='none',
                              s=SCATTER_POINT_SIZE)
    min_val_test = min(y_test.min(), y_pred_test.min())
    max_val_test = max(y_test.max(), y_pred_test.max())
    ax_test.plot([min_val_test, max_val_test], [min_val_test, max_val_test], 'k--', linewidth=1.5)
    cbar_test = plt.colorbar(sc_test, ax=ax_test)
    # <--- Modification: Colorbar label font size increased via global variable
    cbar_test.set_label('Density', fontsize=COLORBAR_LABEL_FONTSIZE)
    ax_test.set_xlabel('Observed Value')
    ax_test.set_ylabel('Predicted Value')
    ax_test.set_title(f'{model_name} (Test Set)')
    metric_text_test = (f"R²: {metrics['test_r2']:.3f}\n"
                        f"RMSE: {metrics['test_rmse']:.3f}\n"
                        f"MAE: {metrics['test_mae']:.3f}\n"
                        f"Pearson’s r: {metrics['test_corr']:.3f}")
    # <--- Modification: Metric text font size increased via global variable
    ax_test.text(0.05, 0.95, metric_text_test, transform=ax_test.transAxes, fontsize=METRIC_FONTSIZE,
                 verticalalignment='top', horizontalalignment='left')
    ax_test.grid(True, alpha=0.3)
    plt.tight_layout()
    if SAVE_PLOTS:
        filename = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_test_scatter.png")
        fig_test.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Test set plot saved: {filename}")
    plt.show()


# =============================================================================
# 3. Define Models and Parameter Grids
# =============================================================================
models_and_params = {
    # TabPFN is used without grid search as it's a non-parametric model
    'TabPFN': {
        'model': TabPFNRegressor(),
        'params': {}
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
    },
    'XGBoost': {
        'model': XGBRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
    },
    'LightGBM': {
        'model': LGBMRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [31, 50]
        }
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'kernel': ['rbf'],
            'C': [1, 10, 100],
            'gamma': ['scale', 'auto']
        }
    },
    'KNN': {
        'model': KNeighborsRegressor(),
        'params': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
    },
    'MLP': {
        'model': MLPRegressor(random_state=42, max_iter=1000),
        'params': {
            'hidden_layer_sizes': [(50, 50), (100,)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001]
        }
    },
    'Ridge': {
        'model': Ridge(),
        'params': {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        }
    },
    'Lasso': {
        'model': Lasso(),
        'params': {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        }
    }
}

# =============================================================================
# 4. Loop for Training, Evaluation, and Plotting
# =============================================================================
results = {}
for model_name, mp in models_and_params.items():
    print(f"\n{'=' * 30}\nProcessing Model: {model_name}\n{'=' * 30}")
    start_time = time.time()
    model_output_dir = os.path.join(BASE_OUTPUT_DIR, model_name.replace(' ', '_'))
    if SAVE_PLOTS and not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    if mp['params']:  # If there are parameters, run GridSearchCV
        grid_search = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='r2', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")
    else:  # If no parameters, just fit the model directly
        best_model = mp['model']
        best_model.fit(X_train, y_train)
        print("Training with default parameters.")

    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_corr': pearsonr(y_train, y_pred_train)[0],
        'test_corr': pearsonr(y_test, y_pred_test)[0],
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
    }
    end_time = time.time()
    metrics['runtime'] = end_time - start_time
    results[model_name] = metrics

    print(f"\n--- {model_name} Results ---")
    print(f"Time elapsed: {metrics['runtime']:.2f} seconds")
    print(f'Training R²: {metrics["train_r2"]:.4f} | Test R²: {metrics["test_r2"]:.4f}')
    print(f'Training RMSE: {metrics["train_rmse"]:.4f} | Test RMSE: {metrics["test_rmse"]:.4f}')
    print(f'Training MAE: {metrics["train_mae"]:.4f} | Test MAE: {metrics["test_mae"]:.4f}')
    print(f'Training Pearson r: {metrics["train_corr"]:.4f} | Test Pearson r: {metrics["test_corr"]:.4f}')

    generate_plots(model_name, y_train, y_pred_train, y_test, y_pred_test, metrics, model_output_dir)

# =============================================================================
# 5. Final Results Summary (Table)
# =============================================================================
print(f"\n{'=' * 40}\n           FINAL MODEL COMPARISON TABLE\n{'=' * 40}")
results_df = pd.DataFrame(results).T
display_cols_order = ['test_r2', 'test_rmse', 'test_mae', 'train_r2', 'train_rmse', 'train_mae', 'runtime']
display_df = results_df[display_cols_order]
print(display_df.sort_values(by='test_r2', ascending=False))


# =============================================================================
# 6. Final Results Summary and Visualization (Bubble Chart)
# =============================================================================
def plot_metrics_bubble_chart(metrics_df, data_type, output_dir):
    """
    Generates a model performance comparison bubble chart for the training or test set.
    The larger the circle, the better the performance.
    """
    plot_df = metrics_df.copy()
    plot_df.reset_index(inplace=True)
    plot_df.rename(columns={'index': 'Model'}, inplace=True)

    r2_col = f'{data_type}_r2'
    plot_df['R²_size'] = plot_df[r2_col].clip(0)  # Clip R2 at 0 for size calculation

    # Normalize error metrics (RMSE, MAE) so that smaller errors -> larger bubbles
    for metric in ['rmse', 'mae']:
        col_name = f'{data_type}_{metric}'
        max_val = plot_df[col_name].max()
        min_val = plot_df[col_name].min()
        if max_val == min_val:
            plot_df[f'{metric.upper()}_size'] = 1.0
        else:
            # Invert the scale: (max - value) / (max - min)
            plot_df[f'{metric.upper()}_size'] = (max_val - plot_df[col_name]) / (max_val - min_val)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor('#F0F0F0')
    ax.set_ylim(0, 1)

    evaluation_metrics = {'R²': 'R²_size', 'RMSE': 'RMSE_size', 'MAE': 'MAE_size'}
    y_positions = [0.8, 0.5, 0.2]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, (metric_name, size_col) in enumerate(evaluation_metrics.items()):
        ax.scatter(plot_df['Model'], [y_positions[i]] * len(plot_df),
                   s=plot_df[size_col] * 1000 + 50,  # Scale bubble size
                   label=metric_name,
                   alpha=0.7,
                   c=colors[i],
                   edgecolors='black',
                   linewidth=0.5)

    # <--- Modification: All font sizes increased by 4
    ax.set_ylabel('Evaluation Metrics', fontsize=18, fontweight='bold')  # was 14
    ax.set_xlabel('Model', fontsize=18, fontweight='bold')  # was 14
    title_str = f'Model Comparison: {data_type.capitalize()} Metrics (Larger bubble is better)'
    ax.set_title(title_str, fontsize=20, fontweight='bold')  # was 16
    ax.set_yticks(y_positions)
    ax.set_yticklabels(evaluation_metrics.keys(), fontweight='bold', fontsize=16)  # was 12
    plt.xticks(rotation=45, ha='right', fontweight='bold', fontsize=16)  # was 12
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    legend = ax.legend(title='Metrics', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=16)  # was 12
    plt.setp(legend.get_title(), fontsize='17', fontweight='bold')  # was 13
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if SAVE_PLOTS:
        filename = os.path.join(output_dir, f"model_comparison_{data_type}.pdf")
        plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=1200)
        print(f"\nBubble chart saved: {filename}")
    plt.show()


plot_metrics_bubble_chart(results_df, 'train', BASE_OUTPUT_DIR)
plot_metrics_bubble_chart(results_df, 'test', BASE_OUTPUT_DIR)

print("\n--- SCRIPT EXECUTION FINISHED ---")