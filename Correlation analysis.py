# این کد میاد ضریب همبستگی رو به ۶ روش محاسبه میکنه
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import dcor
import random

# Load the dataset
file_path = r"C:\(ELIFE)\CHIR\Frequency in KEGG.xlsx"
output_file = r"C:\(ELIFE)\CHIR\correlation_summary.xlsx"

def load_data(file_path, col_x, col_y):
    df = pd.read_excel(file_path, header=0, sheet_name="updated_data_2-20-25")
    # Drop rows with NaN values in the specified columns
    df = df[[col_x, col_y]].dropna()
    return df[col_x], df[col_y]

# Helper to interpret strength (only for linear-style methods)
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
    significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
    interpretation = (f"Pearson measures **linear correlation**.\n"
                      f"Result: {strength} (coefficient = {coeff:.3f}), {significance} (p-value = {p_value:.4f}).")
    return coeff, strength, p_value, significance, interpretation

# Spearman correlation
def spearman_correlation(x, y):
    coeff, p_value = spearmanr(x, y)
    strength = interpret_strength(coeff)
    significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
    interpretation = (f"Spearman measures **monotonic (rank) correlation**.\n"
                      f"Result: {strength} (coefficient = {coeff:.3f}), {significance} (p-value = {p_value:.4f}).")
    return coeff, strength, p_value, significance, interpretation

# Kendall's Tau
def kendall_tau(x, y):
    coeff, p_value = kendalltau(x, y)
    strength = interpret_strength(coeff)
    significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
    interpretation = (f"Kendall's Tau measures **rank association**.\n"
                      f"Result: {strength} (coefficient = {coeff:.3f}), {significance} (p-value = {p_value:.4f}).")
    return coeff, strength, p_value, significance, interpretation

# Maximal Information Coefficient (MIC)
def mic(x, y):
    x = x.values.reshape(-1, 1)
    y = y.values
    mi = mutual_info_regression(x, y, discrete_features=False)[0]
    interpretation = (f"MIC measures **any kind of relationship (linear or nonlinear)**.\n"
                      f"Result: MIC score = {mi:.3f} (higher = stronger association; no p-value available).")
    return mi, None, None, "N/A", interpretation

# Distance Correlation
def distance_correlation(x, y):
    coeff = dcor.distance_correlation(x.values, y.values)
    interpretation = (f"Distance correlation detects **any kind of dependency**.\n"
                      f"Result: Distance correlation = {coeff:.3f} (higher = stronger association; no p-value available).")
    return coeff, None, None, "N/A", interpretation

# Hoeffding's D with permutation-based p-value
def hoeffding_d(x, y, bootstrap_samples=1000):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)

    # Rank the data
    rx = np.argsort(np.argsort(x)) + 1
    ry = np.argsort(np.argsort(y)) + 1

    # Compute Q
    rx_matrix = rx[:, None] < rx
    ry_matrix = ry[:, None] < ry
    Q = np.sum(rx_matrix & ry_matrix) / (n * (n - 1))

    # Compute D
    D = (30 * (n - 2) * (n - 3) * (Q - (n - 1) / 4)) / ((n - 1) * (n - 2) * (n - 3) * (n - 4))

    # Permutation test
    D_permuted = []
    for _ in range(bootstrap_samples):
        y_permuted = np.random.permutation(y)
        ry_perm = np.argsort(np.argsort(y_permuted)) + 1
        ry_matrix_perm = ry_perm[:, None] < ry_perm
        Q_perm = np.sum(rx_matrix & ry_matrix_perm) / (n * (n - 1))
        D_perm = (30 * (n - 2) * (n - 3) * (Q_perm - (n - 1) / 4)) / ((n - 1) * (n - 2) * (n - 3) * (n - 4))
        D_permuted.append(D_perm)

    D_permuted = np.array(D_permuted)
    p_value = np.mean(D_permuted >= D)

    strength = interpret_strength(D)
    significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
    interpretation = (f"Hoeffding's D measures **general dependence** (including non-monotonic relationships).\n"
                      f"Result: {strength} (D = {D:.3f}), {significance} (p-value = {p_value:.4f}).")
    return D, strength, p_value, significance, interpretation

# Main function
def main(file_path, col_x, col_y):
    x, y = load_data(file_path, col_x, col_y)

    # Define methods
    methods = {
        "Pearson Correlation": pearson_correlation,
        "Spearman Correlation": spearman_correlation,
        "Kendall's Tau": kendall_tau,
        "Maximal Information Coefficient (MIC)": mic,
        "Distance Correlation": distance_correlation,
        "Hoeffding's D": hoeffding_d
    }

    results = []

    # Call each method
    for name, func in methods.items():
        coeff, strength, p_value, significance, interpretation = func(x, y)

        results.append({
            "Method": name,
            "Coefficient": round(coeff, 3) if not np.isnan(coeff) else None,
            "Strength": strength if strength is not None else "N/A",
            "p-value": round(p_value, 4) if p_value is not None else "N/A",
            "Significance": significance,
            "Interpretation": interpretation
        })

    # Summary table
    summary_df = pd.DataFrame(results)
    print("\n=== Summary of Correlations ===")
    print(summary_df[["Method", "Coefficient", "Strength", "p-value", "Significance"]].to_string(index=False))

    # Save to Excel
    summary_df.to_excel(output_file, index=False)
    print(f"\nSummary saved to {output_file}")

# Run
main(file_path, "n_chiral_centers", "Assembly Index")
