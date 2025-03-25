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
    plt.savefig('plots_analyze/' + filename)

def det_color(fraudster, external_type):
    if fraudster == 1:
        return 'red'
    if external_type == None or external_type == 'customer':
        return 'green'
    else:
        return dict(
            bank = 'yellow',
            merchant = 'blue')[external_type]

def plot_neighbor_graphs(error_accounts, merged_df, two_accounts_df, n_plots_square, title, filename):
        fraud_map = dict(zip(merged_df['AccountID'], merged_df['Fraudster_true']))
        predicted_map = dict(zip(merged_df['AccountID'], merged_df['Fraudster_pred']))
        
        fig, axs = plt.subplots(n_plots_square, n_plots_square, figsize=(n_plots_square * 3, n_plots_square * 3))
        axs = axs.flatten()
        for idx, acc in enumerate(error_accounts['AccountID']):
            G = nx.Graph()
            G.add_node(acc, fraud=fraud_map.get(acc, 0))
            # Select neighbor transactions; assumes two_accounts_df has columns: AccountID, External, External_Type, Hour, Action, Amount
            neighbors = two_accounts_df[two_accounts_df['AccountID'] == acc]
            for _, row in neighbors.iterrows():
                nb = row['External']
                G.add_node(nb, fraud=fraud_map.get(nb, 0), external_type=row['External_Type'])
                edge_label = f"{row['Hour']}:{row['Action'][:1]}\n {row['Amount']}"
                G.add_edge(acc, nb, label=edge_label)
            pos = nx.spring_layout(G)
            node_colors = [det_color(G.nodes[n].get('fraud', 0) == 1, G.nodes[n].get('external_type')) for n in G.nodes()]
            nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=600, ax=axs[idx])
            edge_labels = nx.get_edge_attributes(G, 'label')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=axs[idx])
            predicted = predicted_map.get(acc, "N/A")
            axs[idx].set_title(f"Acc: {acc} | Pred: {predicted}")

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig('plots_analyze/' + filename)

def plot_transaction_type_distribution(trans_df, accounts_df, merged_df, n_plots_square, title, filename):
    """
    Create pie charts showing transaction distribution by External_Type for each account.
    """
    # Create prediction mapping
    predicted_map = dict(zip(merged_df['AccountID'], merged_df['Fraudster_pred']))
    
    fig, axs = plt.subplots(n_plots_square, n_plots_square, figsize=(n_plots_square*3.5, n_plots_square*3.5))
    axs = axs.flatten()
    
    for idx, acc in enumerate(accounts_df['AccountID']):
        # Get transactions for this account - FIX: Create explicit copy to avoid SettingWithCopyWarning
        acc_trans = trans_df[trans_df['AccountID'] == acc].copy()
        
        if acc_trans.empty:
            axs[idx].text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center')
            axs[idx].set_aspect('equal')
            continue
        
        # Count transactions by External_Type
        # Fill NA with "Cash" to represent cash transactions
        acc_trans['External_Type'] = acc_trans['External_Type'].fillna('Cash')
        type_counts = acc_trans['External_Type'].value_counts()
        
        # Create pie chart with custom colors
        colors = {'Cash': 'lightgreen', 'bank': 'gold', 'merchant': 'skyblue', None: 'gray'}
        color_list = [colors.get(t, 'gray') for t in type_counts.index]
        
        wedges, texts, autotexts = axs[idx].pie(
            type_counts, 
            labels=type_counts.index, 
            autopct='%1.1f%%',
            colors=color_list,
            textprops={'fontsize': 8}
        )
        
        # Adjust font size of percentage text
        for autotext in autotexts:
            autotext.set_fontsize(7)
        
        # Make sure the pie chart is a circle
        axs[idx].set_aspect('equal')
        
        # Add title with account ID and prediction
        predicted = predicted_map.get(acc, "N/A")
        true_label = accounts_df[accounts_df['AccountID'] == acc]['Fraudster_true'].values[0]
        axs[idx].set_title(f"Acc: {acc} | True: {true_label} | Pred: {predicted}", fontsize=8)
    
    # Hide any unused subplots
    for idx in range(len(accounts_df), n_plots_square*n_plots_square):
        axs[idx].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('plots_analyze/' + filename)


def plot_average_transactions_by_type(trans_df, merged_df, title, filename):
    """
    Create a plot showing average number of transactions per account by External_Type,
    comparing fraudulent vs non-fraudulent accounts.
    """
    # Create a copy of the DataFrame
    df = trans_df.copy()
    
    # Fill NA values in External_Type with 'Cash'
    df['External_Type'] = df['External_Type'].fillna('Cash')
    
    # Join with fraud labels
    df = df.merge(merged_df[['AccountID', 'Fraudster_true']], on='AccountID', how='left')
    
    # Group by AccountID and External_Type, count transactions
    transaction_counts = df.groupby(['AccountID', 'External_Type', 'Fraudster_true']).size().reset_index(name='count')
    
    # Calculate average number of transactions per account type (fraud vs non-fraud)
    avg_by_type = transaction_counts.groupby(['External_Type', 'Fraudster_true'])['count'].mean().reset_index()
    
    # Pivot to have Fraud status as columns
    pivot_df = avg_by_type.pivot(index='External_Type', columns='Fraudster_true', values='count')
    pivot_df.columns = ['Non-Fraud', 'Fraud']
    
    # Sort by total transaction count
    pivot_df['Total'] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values('Total', ascending=False)
    pivot_df = pivot_df.drop('Total', axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    pivot_df.plot(kind='bar', figsize=(10, 6))
    plt.title(title)
    plt.xlabel('Transaction Type')
    plt.ylabel('Average Number of Transactions')
    plt.legend(title='Account Type')
    plt.tight_layout()
    plt.savefig('plots_analyze/' + filename)
    
    return pivot_df

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
        plot_neighbor_graphs(error_accounts, merged_df, two_accounts_df, n_plots_square, "Neighbor Graphs for random wrong predictions", "neighbor_graph_subplots.pdf")
        click.echo("Neighbor graphs saved to neighbor_graph_subplots.pdf")

        click.echo("Performing in-depth transaction plots: neighbor graphs")
        # Filter first n_errors error accounts for a grid and sort by true label (non fraudsters first)
        n_plots = min(n_plots_square**2, (~merged_df['error']).sum())
        non_error_accounts = merged_df[~merged_df['error']].sample(n=n_plots, random_state=42)
        non_error_accounts = non_error_accounts.sort_values(by='Fraudster_true', ascending=True)
        plot_neighbor_graphs(non_error_accounts, merged_df, two_accounts_df, n_plots_square, "Neighbor Graphs for random correct predictions", "neighbor_graph_subplots_correct.pdf")
        click.echo("Neighbor graphs saved to neighbor_graph_subplots.pdf")

        click.echo("Plotting transaction type distribution for correct accounts...")
        plot_transaction_type_distribution(
            trans_df, 
            non_error_accounts, 
            merged_df, 
            n_plots_square, 
            "Transaction Type Distribution for Correct Predictions", 
            "transaction_type_distribution_correct.pdf"
        )
        click.echo("Transaction type distribution saved to transaction_type_distribution_correct.pdf")

        # Add average transactions by type plot
        click.echo("Plotting average transactions by type...")
        avg_trans_df = plot_average_transactions_by_type(
            trans_df,
            merged_df,
            "Average Number of Transactions by Type (Fraud vs Non-Fraud)",
            "avg_transactions_by_type.pdf"
        )
        click.echo("Average transactions by type saved to avg_transactions_by_type.pdf")
        click.echo("\nAverage transactions by type summary:")
        click.echo(avg_trans_df)

        # click.echo("Performing in-depth transaction plots: transaction plots")
        # col_plot(trans_df, error_accounts, predicted_map, n_plots_square, ['Amount'], "Amount Heatmaps for random wrong predictions", "transaction_amount_heatmaps.pdf")
        # click.echo("Transaction heatmaps saved to transaction_heatmaps.pdf")
        
        # click.echo("Performing in-depth transaction plots: transaction plots")
        # col_plot(trans_df, error_accounts, predicted_map, n_plots_square, ['OldBalance', 'NewBalance'], "Balance Heatmaps for random wrong predictions", "transaction_balance_heatmaps.pdf")
        # click.echo("Transaction heatmaps saved to transaction_heatmaps.pdf")
        
        # click.echo("Performing in-depth transaction plots: transaction plots")
        # col_plot(trans_df, error_accounts, predicted_map, n_plots_square, ['MissingTransaction'], "Missing Transaction Heatmaps for random wrong predictions", "transaction_missing_heatmaps.pdf")
        # click.echo("Transaction heatmaps saved to transaction_heatmaps.pdf")
        
        

if __name__ == '__main__':
    # example usage
    # python3 analyze_errors.py lvl_account/logs/2025.03.24_15.56.46_attn_cnn_val.csv --n_plots_square 10
    analyze_errors()
