import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import click  # added import
import math  # added import

def plot_heatmap(ax, matrix, x_labels, y_labels, title, cbar=True, annotate=False, annot_values=None, cmap='viridis', **kwargs):
    ax.imshow(matrix, cmap=cmap, **kwargs)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_title(title)
    if annotate:
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                # Changed to use 2D indexing for annot_values
                text = annot_values[i, j] if annot_values is not None else str(matrix[i, j])
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=6)
    if cbar:
        plt.colorbar(ax.images[0], ax=ax)

@click.command()
@click.argument('folder_path', default='lvl_account/logs', type=click.Path(exists=True))
def main(folder_path):
    # Ground truth file path
    ground_truth_path = "/home/yale/Repositories/bbdc25/task/val_set/y_val.ver00.parquet"
    gt_df = pd.read_parquet(ground_truth_path)  # ground truth dataframe (assumes columns: AccountID, Fraudster)
    
    errors = {}   # filename -> set(AccountID) for misclassifications
    corrects = {} # filename -> set(AccountID) for correct predictions
    results_ordered = {}  # filename -> list of tuples (AccountID, is_correct)
    
    # Find all CSV files in folder with _val.csv suffix
    csv_files = list(sorted(glob.glob(os.path.join(folder_path, "*_val.csv"))))
    if not csv_files:
        print("No CSV files with _val.csv suffix found in the provided folder.")
        return
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)  # predictions dataframe (assumes columns: AccountID, Fraudster)
        # Merge with ground truth: suffixes _pred and _true.
        merged = pd.merge(df, gt_df, on="AccountID", suffixes=('_pred', '_true'))
        # Identify wrong predictions: where Fraudster_pred != Fraudster_true
        wrong = merged.loc[merged['Fraudster_pred'] != merged['Fraudster_true'], 'AccountID']
        errors[os.path.basename(csv_file)] = set(wrong)
        # Compute correct predictions
        correct = merged.loc[merged['Fraudster_pred'] == merged['Fraudster_true'], 'AccountID']
        corrects[os.path.basename(csv_file)] = set(correct)
        # Save ordered results for individual heatmaps
        order = merged["AccountID"].tolist()
        is_correct = (merged['Fraudster_pred'] == merged['Fraudster_true']).tolist()
        results_ordered[os.path.basename(csv_file)] = list(zip(order, is_correct))
    
    file_names = list(errors.keys())
    n = len(file_names)
    
    # Compute overlap and difference matrices
    overlap_matrix = np.zeros((n, n), dtype=int)
    diff_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            set_i = errors[file_names[i]]
            set_j = errors[file_names[j]]
            overlap_matrix[i, j] = len(set_i & set_j)
            diff_matrix[i, j] = len(set_i ^ set_j)
    
    # Plot overlap and difference heatmaps using helper function
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    plot_heatmap(axs[0], overlap_matrix, file_names, file_names,
                 "Overlap of Wrong Predictions (Intersection Size)",
                 annotate=True, cmap='Blues')
    plot_heatmap(axs[1], diff_matrix, file_names, file_names,
                 "Differences in Wrong Predictions (Symmetric Difference Size)",
                 annotate=True, cmap='Reds')
    plt.tight_layout()
    plt.savefig('plots_analyze/predictions_comparison.pdf')
    print('plots_analyze/predictions_comparison.pdf')
    
    # Meta-Model Analysis:
    # For each model, compute rescued errors (misclassified samples that are correct in at least one other model)
    rescued_counts = {}
    for model in file_names:
        other_correct = set().union(*(corrects[m] for m in file_names if m != model))
        rescued = errors[model] & other_correct
        rescued_counts[model] = len(rescued)
    
    # Visualization: Bar chart comparing total errors vs rescued errors per model
    total_errors = [len(errors[m]) for m in file_names]
    rescued_errors = [rescued_counts[m] for m in file_names]
    x = np.arange(len(file_names))
    width = 0.35
    fig2, ax2 = plt.subplots(figsize=(10,6))
    ax2.bar(x - width/2, total_errors, width, label='Total Errors', color='gray')
    ax2.bar(x + width/2, rescued_errors, width, label='Rescued Errors', color='green')
    ax2.set_xticks(x)
    ax2.set_xticklabels(file_names, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Count")
    ax2.set_title("Meta-Model Analysis: Errors Rescued by Other Models")
    ax2.legend()
    plt.tight_layout()
    plt.savefig('plots_analyze/rescued_errors_comparison.pdf')
    print('plots_analyze/rescued_errors_comparison.pdf')

    # New section: Non-Rescuable Errors Heatmap
    overall_errors = set().union(*[errors[m] for m in file_names])
    overall_correct = set().union(*[corrects[m] for m in file_names])
    non_rescuable_ids = overall_errors - overall_correct

    # create a pred csv so that we can ingest the non-rescuable errors into other analysis tools
    non_rescuable_df = gt_df.copy()
    # overwrite the Fraudster column with the comp value ie 0 -> 1, 1 -> 0 for each account in non_rescuable_ids
    non_rescuable_df.loc[non_rescuable_df['AccountID'].isin(non_rescuable_ids), 'Fraudster'] = 1 - non_rescuable_df['Fraudster'] 
    non_rescuable_df.to_csv('non_rescuable_errors.csv', index=False)
    print('non_rescuable_errors.csv')

    # Sort AccountIDs and build lists for true labels and annotations (AccountIDs)
    non_rescuable_ids_sorted = sorted([acc for acc in non_rescuable_ids if acc in gt_df['AccountID'].values])
    true_labels = [gt_df[gt_df['AccountID'] == acc]['Fraudster'].values[0] for acc in non_rescuable_ids_sorted]
    
    num_cols = 10  # fixed number of columns
    n = len(true_labels)
    num_rows = math.ceil(n / num_cols)
    padding = (num_rows * num_cols) - n
    true_labels.extend([np.nan] * padding)
    annot_account_ids = non_rescuable_ids_sorted + [''] * padding

    heatmap_matrix = np.array(true_labels).reshape(num_rows, num_cols)
    annot_values_matrix = np.array(annot_account_ids).reshape(num_rows, num_cols)
    
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    plot_heatmap(ax3, heatmap_matrix, 
                 x_labels=[str(i) for i in range(num_cols)], 
                 y_labels=[str(i) for i in range(num_rows)],
                 title="Non-Rescuable Errors True Labels", 
                 annotate=True, 
                 annot_values=annot_values_matrix,
                 cmap='coolwarm',
                 vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig('plots_analyze/non_rescuable_errors_heatmap.pdf')
    print('plots_analyze/non_rescuable_errors_heatmap.pdf')

if __name__ == "__main__":
    main()  # using click to handle command-line arguments