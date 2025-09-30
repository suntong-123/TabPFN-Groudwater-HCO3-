# ==============================================================================
# PART 0: Import All Necessary Libraries
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
import warnings
import time

# Feature selection tools
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Model evaluation and pipeline tools
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# TabPFN
try:
    from tabpfn import TabPFNRegressor
except ImportError:
    print("Warning: TabPFNRegressor not found. Please ensure the tabpfen library is installed correctly.")
    TabPFNRegressor = None

# Suppress specific warnings
warnings.filterwarnings('ignore', message='Running on CPU with more than 200 samples may be slow.*')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='tabpfen')


# ==============================================================================
# PART 1: Feature Selection and Helper Functions (Keep as is)
# ==============================================================================

def intelligent_vif_selection(X: pd.DataFrame, y: pd.Series, vif_threshold: float = 10.0):
    """Stage 1.1: Intelligent VIF Selection"""
    print("--- [1.1] Starting Intelligent VIF Feature Selection ---")
    features_to_keep = X.columns.tolist()
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns, name="initial_importance_for_vif")
    iteration = 0
    while True:
        iteration += 1
        if len(features_to_keep) <= 2: break
        X_current = X[features_to_keep]
        X_vif_calc = add_constant(X_current.values)
        try:
            vif_values = pd.Series([variance_inflation_factor(X_vif_calc, i) for i in range(1, X_vif_calc.shape[1])],
                                   index=features_to_keep)
        except np.linalg.LinAlgError:
            print("  Warning: Singular matrix encountered in VIF calculation, possibly due to perfect multicollinearity. Skipping this VIF check.")
            break
        high_vif_features = vif_values[vif_values > vif_threshold]
        if high_vif_features.empty:
            print(f"  All remaining features have VIF values below {vif_threshold}, VIF selection complete.")
            break
        feature_to_drop = importances.loc[high_vif_features.index].idxmin()
        features_to_keep.remove(feature_to_drop)
        print(
            f"  VIF Iteration {iteration}: Dropping feature '{feature_to_drop}' (VIF: {high_vif_features.loc[feature_to_drop]:.2f}, lowest importance in high VIF group)")
    print(f"--- [1.1] VIF selection finished, {len(features_to_keep)} features remaining ---")
    return features_to_keep, importances


def pairwise_correlation_selection(X: pd.DataFrame, importances: pd.Series, corr_threshold: float = 0.8):
    """Stage 1.2: Pairwise Correlation Refinement"""
    print("\n--- [1.2] Starting Pairwise High Correlation Feature Refinement ---")
    features_to_keep = X.columns.tolist()
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = upper_tri.stack().sort_values(ascending=False)
    high_corr_pairs = high_corr_pairs[high_corr_pairs > corr_threshold]
    to_drop = set()
    if high_corr_pairs.empty:
        print(f"  No feature pairs found with absolute correlation exceeding {corr_threshold}.")
    else:
        print(f"  Found {len(high_corr_pairs)} pairs of features with correlation > {corr_threshold}, processing...")
        for (feat1, feat2), corr_val in high_corr_pairs.items():
            if feat1 not in to_drop and feat2 not in to_drop:
                importance1 = importances.get(feat1, 0)
                importance2 = importances.get(feat2, 0)
                if importance1 >= importance2:
                    feature_to_drop = feat2
                else:
                    feature_to_drop = feat1
                to_drop.add(feature_to_drop)
                print(f"  - Comparing '{feat1}' and '{feat2}' (Correlation: {corr_val:.2f})")
                print(
                    f"    Importance: '{feat1}'({importance1:.4f}) vs '{feat2}'({importance2:.4f}). Decided to drop '{feature_to_drop}'.")
    final_features = [f for f in features_to_keep if f not in to_drop]
    print(f"--- [1.2] Refinement finished, {len(final_features)} features remaining ---")
    return final_features


def get_comprehensive_ranking(X: pd.DataFrame, y: pd.Series):
    """Stage 2: Borda Comprehensive Ranking"""
    print("\n--- [2.0] Starting Comprehensive Feature Ranking (Borda Count) ---")
    feature_names = X.columns.tolist()
    num_features = len(feature_names)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    ranking_df = pd.DataFrame({'feature': feature_names})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ranking_df['rf_importance_score'] = rf.feature_importances_
        ranking_df['pearson_correlation'] = X.corrwith(y).abs().fillna(0).values
        ranking_df['mutual_information'] = mutual_info_regression(X, y, random_state=42)
    rank_rf = ranking_df.sort_values(by='rf_importance_score', ascending=False)['feature'].tolist()
    rank_pearson = ranking_df.sort_values(by='pearson_correlation', ascending=False)['feature'].tolist()
    rank_mi = ranking_df.sort_values(by='mutual_information', ascending=False)['feature'].tolist()
    borda_scores = {feature: 0 for feature in feature_names}
    for rank_list in [rank_rf, rank_pearson, rank_mi]:
        for i, feature in enumerate(rank_list):
            borda_scores[feature] += (num_features - i)
    ranking_df['borda_score'] = ranking_df['feature'].map(borda_scores)
    ranking_df = ranking_df.sort_values(by='borda_score', ascending=False).reset_index(drop=True)
    ranking_df['borda_rank'] = ranking_df.index + 1
    print("  Borda comprehensive ranking has been generated.")
    return ranking_df


def check_distribution_consistency(train_data: pd.Series, test_data: pd.Series, name: str):
    """Compare the distribution of a variable on the training and test sets"""
    plt.figure(figsize=(10, 4))
    sns.kdeplot(train_data, label='Train', color='blue', shade=True)
    sns.kdeplot(test_data, label='Test', color='red', shade=True)
    plt.title(f'Distribution of "{name}" in Train vs. Test set', fontsize=14);
    plt.legend();
    plt.show()
    stats = pd.DataFrame({'Train': train_data.describe(), 'Test': test_data.describe()})
    print(f"\nStatistical Summary for '{name}':\n{stats}")
    ks_stat, p_value = ks_2samp(train_data.dropna(), test_data.dropna())
    print(f"\nKolmogorov-Smirnov Test for '{name}':")
    print(f"  KS-Statistic: {ks_stat:.4f}, P-value: {p_value:.4f}")
    if p_value > 0.05:
        print("  -> Conclusion: P-value > 0.05, distributions are basically consistent.")
    else:
        print("  -> Warning: P-value <= 0.05, distributions are significantly different!")


# --- [Core Modification] Forward Selection Function ---
def run_forward_selection(X: pd.DataFrame, y: pd.Series, candidate_features: list, initial_features: list,
                          max_features_to_select: int):
    """Stage 3: Forward Selection, and return performance history"""
    print("\n--- [3.0] Starting Forward Selection ---")

    # [Modification 1] Change scoring metric to r2
    scoring_metric = 'r2'
    performance_history, feature_sets_history = [], []
    current_features = initial_features.copy()

    selection_pipeline = Pipeline([('scaler', StandardScaler()), ('model', TabPFNRegressor(device='cpu'))])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"  Calculating baseline performance (using initial {len(current_features)} features)...")
    X_current = X[current_features]
    initial_score = np.mean(cross_val_score(selection_pipeline, X_current, y, cv=cv, scoring=scoring_metric, n_jobs=-1))
    performance_history.append(initial_score)
    feature_sets_history.append(current_features.copy())
    # [Modification 1] Update print message
    print(f"  Initial features: {current_features}, Initial Cross-Validation R²: {initial_score:.4f}")

    for i in range(len(initial_features), max_features_to_select):
        iteration_start_time = time.time()
        print(f"\n  --- Iteration: Finding the {i + 1}-th best feature ---")
        remaining_features = [f for f in candidate_features if f not in current_features]
        if not remaining_features:
            print("    No more features to select, stopping iteration.")
            break
        best_candidate, best_score = None, -np.inf
        for idx, candidate in enumerate(remaining_features):
            print(f"    ({idx + 1}/{len(remaining_features)}) Testing: {candidate} ...", end="", flush=True)
            temp_features = current_features + [candidate]
            X_subset = X[temp_features]
            scores = cross_val_score(selection_pipeline, X_subset, y, cv=cv, scoring=scoring_metric, n_jobs=-1)
            current_score = np.mean(scores)
            # [Modification 1] Update print message
            print(f" CV R²: {current_score:.4f}")
            if current_score > best_score:
                best_score, best_candidate = current_score, candidate
        if best_candidate is None:
            print("    No feature found that improves performance, stopping selection.")
            break
        current_features.append(best_candidate)
        performance_history.append(best_score)
        feature_sets_history.append(current_features.copy())
        iteration_end_time = time.time()
        # [Modification 1] Update print message
        print(f"  -> Winner of this round: '{best_candidate}', Model CV R² improved to: {best_score:.4f}")
        print(f"  Time for this round: {iteration_end_time - iteration_start_time:.2f} seconds")
    print("\n--- [3.0] Forward selection process finished ---")
    return performance_history, feature_sets_history


# ==============================================================================
# PART 2: Main Workflow
# ==============================================================================
if __name__ == '__main__':
    # --- Steps A, B, B.1 Keep as is ---
    print("=" * 80 + "\nPART A: Data Loading and Preparation\n" + "=" * 80)
    file_path = r"C:/Users/13600K/Desktop/中国预测/全部变量 - 副本.csv"
    target_col = "HCO3"
    try:
        df = pd.read_csv(file_path, encoding='utf-8');
        df.columns = df.columns.str.strip()
        df_numeric = df.select_dtypes(include=np.number).dropna()
        print(f"File loaded successfully. Dataset size after preprocessing: {df_numeric.shape}")
    except (FileNotFoundError, ImportError):
        print(f"Error: Demo data will be used.")
        X_demo = pd.DataFrame(np.random.rand(200, 25), columns=[f'feature_{i}' for i in range(25)])
        y_demo = pd.Series(2 * X_demo['feature_0'] + np.random.randn(200), name='HCO3')
        df_numeric = pd.concat([X_demo, y_demo], axis=1)
    X_full, y_full = df_numeric.drop(target_col, axis=1), df_numeric[target_col]

    print("\n" + "=" * 80 + "\nPART B: Splitting into Training and Test Sets\n" + "=" * 80)
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    print(f"Dataset has been split into 80% training set ({X_train.shape[0]} samples) and 20% test set ({X_test.shape[0]} samples).")

    print("\n" + "=" * 80 + "\nPART B.1: Verifying Distribution Consistency between Training and Test Sets\n" + "=" * 80)
    check_distribution_consistency(y_train, y_test, target_col)
    if X_train.shape[1] > 3:
        for feature in X_train.columns.to_series().sample(3, random_state=42):
            check_distribution_consistency(X_train[feature], X_test[feature], feature)

    # --- Step C Keep as is ---
    print("\n" + "=" * 80 + "\nPART C: Executing Feature Selection Process on the Training Set\n" + "=" * 80)
    features_after_vif, initial_importances = intelligent_vif_selection(X_train, y_train)
    X_train_after_vif = X_train[features_after_vif]
    features_after_corr = pairwise_correlation_selection(X_train_after_vif, initial_importances)
    X_train_screened = X_train[features_after_corr]
    ranking_df = get_comprehensive_ranking(X_train_screened, y_train)
    print(
        "\nBorda Comprehensive Ranking Results (Top 15):\n" + ranking_df[['borda_rank', 'feature', 'borda_score']].head(15).to_string())

    # --- Calling part of Step C, with core modifications ---
    ranked_list = ranking_df['feature'].tolist()
    # [Modification 2] Initial feature set contains only the top-ranked feature from Borda
    initial_features = ranked_list[:1]
    max_features = min(15, len(ranked_list))

    perf_history, fsets_history = run_forward_selection(
        X_train, y_train,
        candidate_features=ranked_list,
        initial_features=initial_features,
        max_features_to_select=max_features
    )

    # --- Step D: Analysis and Plotting, with core modifications ---
    print("\n" + "=" * 80 + "\nPART D: Determining the Optimal Feature Set\n" + "=" * 80)

    # [Modification 1 & 2] Analysis logic updated to R²
    num_features_axis = list(range(len(initial_features), len(perf_history) + len(initial_features)))
    r2_history = perf_history  # Directly use performance history as it is now R²
    best_r2_index = np.argmax(r2_history)  # Find the index of the maximum R²
    best_num_features = num_features_axis[best_r2_index]
    best_r2_value = r2_history[best_r2_index]
    final_selected_features = fsets_history[best_r2_index]

    print("Automated Analysis Results:")
    print(f"  -> The best cross-validation performance was achieved with {best_num_features} features.")
    print(f"  -> The highest cross-validation R² is: {best_r2_value:.4f}")
    print("  -> The recommended optimal feature set is:")
    for i, feature in enumerate(final_selected_features): print(f"    {i + 1}. {feature}")

    # [Modification 1] Plotting updated to R²
    plt.figure(figsize=(12, 7));
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.plot(num_features_axis, r2_history, marker='o', linestyle='-', color='g', label='5-Fold Cross-Validation R²')
    plt.xlabel("Number of Features", fontsize=12);
    plt.ylabel("R² Score", fontsize=12)
    plt.title("Forward Selection Process: Number of Features vs. Model Performance (R²)", fontsize=14, weight='bold')
    plt.xticks(num_features_axis)
    plt.plot(best_num_features, best_r2_value, 'r*', markersize=15,
             label=f'Optimal Point: {best_num_features} features, R²={best_r2_value:.4f}')
    plt.legend(fontsize=12);
    plt.show()

    # --- Step E Keep as is ---
    print("\n" + "=" * 80 + "\nPART E: Evaluating the Final Model on the Test Set\n" + "=" * 80)
    if TabPFNRegressor is None: exit("TabPFNRegressor is not available, skipping final evaluation.")
    print("Using the recommended optimal feature set to perform final evaluation on the [Test Set]...")
    X_train_final, X_test_final = X_train[final_selected_features], X_test[final_selected_features]
    final_pipeline = Pipeline([('scaler', StandardScaler()), ('model', TabPFNRegressor(device='cpu'))])

    print("Training the final model...");
    final_pipeline.fit(X_train_final, y_train)
    y_pred_test = final_pipeline.predict(X_test_final)

    test_r2, test_rmse, test_mae = r2_score(y_test, y_pred_test), np.sqrt(
        mean_squared_error(y_test, y_pred_test)), mean_absolute_error(y_test, y_pred_test)

    print("\nFinal Model Performance Evaluation on the [Test Set]:")
    print(f'  R² Score: {test_r2:.4f}');
    print(f'  RMSE:     {test_rmse:.4f}');
    print(f'  MAE:      {test_mae:.4f}\n')

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_test, alpha=0.7, edgecolors='k', s=60, label='Predicted vs. Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='1:1 Line')
    plt.xlabel('Actual Values', fontsize=12);
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'Final Model Performance on Test Set (R² = {test_r2:.4f})', fontsize=14, weight='bold')
    plt.legend();
    plt.grid(True);
    plt.show()

    print("\n--- All processes completed ---")