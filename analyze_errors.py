import click
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

def col_plot(trans_df, error_accounts, predicted_map, n_plots_square, cols, title, filename):
    # Compute global vmin and vmax for the selected accounts (i.e., share the z-axis)
    df_global = trans_df[trans_df['AccountID'].isin(error_accounts['AccountID'])][cols]
    global_vmin = df_global.min().min()
    global_vmax = df_global.max().max()

    fig2, axs2 = plt.subplots(n_plots_square, n_plots_square, figsize=(n_plots_square*3, n_plots_square*3), sharex=True)
    axs2 = axs2.flatten()
    for idx, acc in enumerate(error_accounts['AccountID']):
        df_plot = trans_df[trans_df['AccountID'] == acc][cols]
        if df_plot.empty:
            axs2[idx].text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center')
        else:
            sns.heatmap(df_plot, ax=axs2[idx], cbar=True, vmin=global_vmin, vmax=global_vmax)
        predicted = predicted_map.get(acc, "N/A")
        axs2[idx].set_title(f"Acc: {acc} | Pred: {predicted}")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)

@click.command()
# @click.argument('pred_csv', default="lvl_account/logs/2025.03.24_11.06.32_attn_cnn_val.csv", type=click.Path(exists=True))
@click.argument('pred_csv', type=click.Path(exists=True))
@click.argument('true_parquet', default='task/val_set/y_val.parquet', type=click.Path(exists=True))
@click.option('--transaction_file', type=click.Path(exists=True), default='task/val_set/x_val.ver08.parquet', help='CSV file with transaction data for in-depth analysis')
@click.option('--n_plots_square', type=int, default=4, help='Number of error accounts to plot')
def analyze_errors(pred_csv, true_parquet, transaction_file, n_plots_square):
    # Load CSV files for true labels and predictions
    true_df = pd.read_parquet(true_parquet)
    pred_df = pd.read_csv(pred_csv)
    
    # Merge data on AccountID
    merged_df = pd.merge(true_df, pred_df, on="AccountID", suffixes=('_true', '_pred'))
    
    # Compute error analysis using Fraudster columns
    merged_df['error'] = merged_df['Fraudster_true'] != merged_df['Fraudster_pred']
    total = len(merged_df)
    errors = merged_df['error'].sum()
    accuracy = (total - errors) / total if total > 0 else 0
    
    click.echo(f"Total records: {total}")
    click.echo(f"Number of errors: {errors}")
    click.echo(f"Accuracy: {accuracy:.2%}")
    
    # In-depth analysis using the transaction file if provided
    if transaction_file:
        trans_df = pd.read_parquet(transaction_file)
        two_accounts_df = trans_df[trans_df['External'].notna()].copy()

        click.echo("Performing in-depth transaction plots: neighbor graphs")
        # Filter first n_errors error accounts for a grid and sort by true label (non fraudsters first)
        n_errors = min(n_plots_square**2, merged_df['error'].sum())
        error_accounts = merged_df[merged_df['error']].sample(n=n_errors, random_state=42)
        error_accounts = error_accounts.sort_values(by='Fraudster_true', ascending=True)
        fraud_map = dict(zip(merged_df['AccountID'], merged_df['Fraudster_true']))
        predicted_map = dict(zip(merged_df['AccountID'], merged_df['Fraudster_pred']))
        
        fig, axs = plt.subplots(n_plots_square, n_plots_square, figsize=(n_plots_square*3, n_plots_square*3))
        axs = axs.flatten()
        for idx, acc in enumerate(error_accounts['AccountID']):
            G = nx.Graph()
            G.add_node(acc, fraud=fraud_map.get(acc, 0))
            # Select up to 3 neighbor transactions; assume two_accounts_df has columns: AccountID, External, Hour, Action, Amount
            neighbors = two_accounts_df[two_accounts_df['AccountID'] == acc].head(3)
            for _, row in neighbors.iterrows():
                nb = row['External']
                G.add_node(nb, fraud=fraud_map.get(nb, 0))
                edge_label = f"{row['Hour']}:{row['Action'][:1]}\n {row['Amount']}"
                G.add_edge(acc, nb, label=edge_label)
            pos = nx.spring_layout(G)
            node_colors = ['red' if G.nodes[n].get('fraud', 0)==1 else 'green' for n in G.nodes()]
            nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=600, ax=axs[idx])
            edge_labels = nx.get_edge_attributes(G, 'label')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=axs[idx])
            predicted = predicted_map.get(acc, "N/A")
            axs[idx].set_title(f"Acc: {acc} | Pred: {predicted}")

        plt.suptitle("Neighbor Graphs for random wrong predictions")
        plt.tight_layout()
        plt.savefig("neighbor_graph_subplots.pdf")
        click.echo("Neighbor graphs saved to neighbor_graph_subplots.pdf")

        click.echo("Performing in-depth transaction plots: transaction plots")
        col_plot(trans_df, error_accounts, predicted_map, n_plots_square, ['Amount'], "Amount Heatmaps for random wrong predictions", "transaction_amount_heatmaps.pdf")
        click.echo("Transaction heatmaps saved to transaction_heatmaps.pdf")
        
        click.echo("Performing in-depth transaction plots: transaction plots")
        col_plot(trans_df, error_accounts, predicted_map, n_plots_square, ['OldBalance', 'NewBalance'], "Balance Heatmaps for random wrong predictions", "transaction_balance_heatmaps.pdf")
        click.echo("Transaction heatmaps saved to transaction_heatmaps.pdf")
        
        click.echo("Performing in-depth transaction plots: transaction plots")
        col_plot(trans_df, error_accounts, predicted_map, n_plots_square, ['MissingTransaction'], "Missing Transaction Heatmaps for random wrong predictions", "transaction_missing_heatmaps.pdf")
        click.echo("Transaction heatmaps saved to transaction_heatmaps.pdf")
        
        

if __name__ == '__main__':
    analyze_errors()
