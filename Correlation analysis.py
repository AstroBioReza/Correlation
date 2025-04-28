import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import dcor
import random

# Load the dataset from an Excel file
file_path = r"D:\(ELIFE)\CHIR\updated_data_3-7-25.xlsx"
output_file = r"D:\(ELIFE)\CHIR\correlation_summary.xlsx"
def load_data(file_path, col_x, col_y):
    df = pd.read_excel(file_path, header=0, sheet_name="Blanks_deleted")
    return df[col_x], df[col_y]

# Helper to interpret correlation strength
def interpret_strength(coeff):
    abs_coeff = abs(coeff)
    if abs_coeff < 0.1:
        return "very weak or no correlation"
    elif abs_coeff < 0.3:
        return "weak correlation"
    elif abs_coeff < 0.5:
        return "moderate correlation"
    elif abs_coeff < 0.7:
        return "strong correlation"
    else:
        return "very strong correlation"

# Pearson correlation
def pearson_correlation(x, y):
    coeff, p_value = pearsonr(x, y)
    strength = interpret_strength(coeff)
    if p_value < 0.05:
        significance = "statistically significant"
    else:
        significance = "not statistically significant"
    interpretation = (f"Pearson measures **linear correlation**.\n"
                      f"Result: {strength} (coefficient = {coeff:.3f}), {significance} (p-value = {p_value:.4f}).")
    return coeff, interpretation

# Spearman correlation
def spearman_correlation(x, y):
    coeff, p_value = spearmanr(x, y)
    strength = interpret_strength(coeff)
    if p_value < 0.05:
        significance = "statistically significant"
    else:
        significance = "not statistically significant"
    interpretation = (f"Spearman measures **monotonic (rank) correlation**.\n"
                      f"Result: {strength} (coefficient = {coeff:.3f}), {significance} (p-value = {p_value:.4f}).")
    return coeff, interpretation

# Kendall's Tau
def kendall_tau(x, y):
    coeff, p_value = kendalltau(x, y)
    strength = interpret_strength(coeff)
    if p_value < 0.05:
        significance = "statistically significant"
    else:
        significance = "not statistically significant"
    interpretation = (f"Kendall's Tau measures **rank association**.\n"
                      f"Result: {strength} (coefficient = {coeff:.3f}), {significance} (p-value = {p_value:.4f}).")
    return coeff, interpretation

# Maximal Information Coefficient (MIC)
def mic(x, y):
    x = x.values.reshape(-1, 1)  # Reshape for sklearn
    y = y.values
    mi = mutual_info_regression(x, y, discrete_features=False)
    strength = interpret_strength(mi[0])
    interpretation = (f"MIC measures **any kind of relationship (linear or nonlinear)**.\n"
                      f"Result: {strength} (MIC score = {mi[0]:.3f}).\n"
                      f"Note: No p-value is available, so interpret with caution.")
    return mi[0], interpretation

# Distance Correlation (NEW)
def distance_correlation(x, y):
    """
    Distance correlation detects **any association** (linear, nonlinear, complex).
    dcor.distance_correlation returns 0 if independent, closer to 1 if dependent.
    """
    coeff = dcor.distance_correlation(x.values, y.values)
    strength = interpret_strength(coeff)
    interpretation = (f"Distance correlation detects **any kind of dependency**.\n"
                      f"Result: {strength} (Distance correlation = {coeff:.3f}).\n"
                      f"No p-value by default; high value means stronger relation.")
    return coeff, interpretation

def hoeffding_d(x, y, bootstrap_samples=1000):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)

    # Step 1: Rank the data
    rx = np.argsort(np.argsort(x)) + 1
    ry = np.argsort(np.argsort(y)) + 1

    # Step 2: Calculate Q
    Q = np.sum([(rx[i] < rx[j]) & (ry[i] < ry[j]) for i in range(n) for j in range(n) if i != j])
    Q = Q / (n * (n - 1))

    # Step 3: Calculate D
    D = (30 * (n - 2) * (n - 3) * (Q - (n - 1) / 4)) / ((n - 1) * (n - 2) * (n - 3) * (n - 4))

    # Bootstrap procedure
    D_bootstrap = []
    for _ in range(bootstrap_samples):
        idx = np.random.choice(n, size=n, replace=True)
        x_resampled = x[idx]
        y_resampled = y[idx]
        
        rx_resampled = np.argsort(np.argsort(x_resampled)) + 1
        ry_resampled = np.argsort(np.argsort(y_resampled)) + 1

        Q_resampled = np.sum([(rx_resampled[i] < rx_resampled[j]) & (ry_resampled[i] < ry_resampled[j]) 
                              for i in range(n) for j in range(n) if i != j])
        Q_resampled = Q_resampled / (n * (n - 1))

        D_resampled = (30 * (n - 2) * (n - 3) * (Q_resampled - (n - 1) / 4)) / ((n - 1) * (n - 2) * (n - 3) * (n - 4))
        D_bootstrap.append(D_resampled)
    
    p_value = np.mean(np.array(D_bootstrap) >= D)
    strength = interpret_strength(D)
    significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
    
    return D, strength, p_value, significance

# Main function to calculate and print all correlations
def main(file_path, col_x, col_y):
    x, y = load_data(file_path, col_x, col_y)

    # Collect results
    results = []

    methods = {
        "Pearson Correlation": pearson_correlation(x, y),
        "Spearman Correlation": spearman_correlation(x, y),
        "Kendall's Tau": kendall_tau(x, y),
        "Maximal Information Coefficient (MIC)": mic(x, y),
        "Distance Correlation": distance_correlation(x, y),
        "Hoeffding's D": hoeffding_d(x, y)
    }

    for name, func in methods.items():
        coeff, strength, p_value, significance = func(x, y)
        results.append({
            "Method": name,
            "Coefficient": round(coeff, 3) if not np.isnan(coeff) else None,
            "Strength": strength,
            "p-value": round(p_value, 4) if p_value is not None else None,
            "Significance": significance
        })

    # Create a summary table
    summary_df = pd.DataFrame(results)
    print("\n=== Summary of Correlations ===")
    print(summary_df.to_string(index=False))

    # Save the summary table to Excel
    output_file = "correlation_summary.xlsx"
    summary_df.to_excel(output_file, index=False)
    print(f"\nSummary saved to {output_file}")

# Run the code
main(file_path, "n_chiral_centers", "Assembly Index")
