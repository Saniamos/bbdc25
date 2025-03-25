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

def plot_neighbor_graphs(error_accounts, predicted_map, merged_df, two_accounts_df, n_plots_square, title, filename):
        fraud_map = dict(zip(merged_df['AccountID'], merged_df['Fraudster_true']))
        
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
    
    Args:
        trans_df: DataFrame with transaction data
        accounts_df: DataFrame with accounts to plot
        merged_df: DataFrame with predictions and true labels
        n_plots_square: Number of plots per row/column
        title: Plot title
        filename: Output filename
    """
    # Create prediction mapping
    predicted_map = dict(zip(merged_df['AccountID'], merged_df['Fraudster_pred']))
    
    fig, axs = plt.subplots(n_plots_square, n_plots_square, figsize=(n_plots_square*3.5, n_plots_square*3.5))
    axs = axs.flatten()
    
    for idx, acc in enumerate(accounts_df['AccountID']):
        # Get transactions for this account
        acc_trans = trans_df[trans_df['AccountID'] == acc]
        
        if acc_trans.empty:
            axs[idx].text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center')
            axs[idx].set_aspect('equal')
            continue
        
        # Count transactions by External_Type
        # Fill NA with "Cash" to represent cash transactions using .loc to avoid a SettingWithCopyWarning
        acc_trans.loc[:, 'External_Type'] = acc_trans['External_Type'].fillna('Cash')
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

def plot_external_partner_frequency(trans_df, accounts_df, merged_df, n_plots_square, title, filename):
    """
    Create histograms showing transaction frequencies with external partners for error accounts.
    Highlight external partners that appear in multiple accounts.
    """
    # Create prediction mapping
    predicted_map = dict(zip(merged_df['AccountID'], merged_df['Fraudster_pred']))
    
    # First pass: collect all external partners and count which accounts they appear in
    external_partners_count = {}
    account_transactions = {}
    
    for acc in accounts_df['AccountID']:
        # Get transactions for this account with external partners
        acc_trans = trans_df[(trans_df['AccountID'] == acc) & (trans_df['External'].notna())].copy()
        account_transactions[acc] = acc_trans
        
        if not acc_trans.empty:
            # Get unique external partners for this account
            unique_partners = acc_trans['External'].unique()
            for partner in unique_partners:
                if partner in external_partners_count:
                    external_partners_count[partner].add(acc)
                else:
                    external_partners_count[partner] = {acc}
    
    # Determine which partners appear in multiple accounts
    shared_partners = {partner for partner, accounts in external_partners_count.items() 
                      if len(accounts) > 1}
    
    # Create the plots
    fig, axs = plt.subplots(n_plots_square, n_plots_square, figsize=(n_plots_square*4, n_plots_square*3.5))
    axs = axs.flatten()
    
    for idx, acc in enumerate(accounts_df['AccountID']):
        acc_trans = account_transactions[acc]
        
        if acc_trans.empty or len(acc_trans['External'].unique()) <= 1:
            axs[idx].text(0.5, 0.5, "No external transactions", horizontalalignment='center', verticalalignment='center')
            continue
            
        # Count transactions by external partner
        partner_counts = acc_trans['External'].value_counts()
        
        # Get transaction type for coloring
        acc_trans['External_Type'] = acc_trans['External_Type'].fillna('Cash')
        partner_types = {ext: acc_trans[acc_trans['External'] == ext]['External_Type'].iloc[0] 
                        for ext in partner_counts.index}
        
        # Define colors for partner types
        colors = {'Cash': 'lightgreen', 'bank': 'gold', 'merchant': 'skyblue', None: 'gray'}
        
        # Create histogram with custom colors
        bars = axs[idx].bar(
            range(len(partner_counts)), 
            partner_counts.values,
            color=[colors.get(partner_types.get(p), 'gray') for p in partner_counts.index],
            edgecolor=['red' if p in shared_partners else 'black' for p in partner_counts.index],
            linewidth=[2 if p in shared_partners else 1 for p in partner_counts.index]
        )
        
        # Add labels
        true_label = accounts_df[accounts_df['AccountID'] == acc]['Fraudster_true'].values[0]
        predicted = predicted_map.get(acc, "N/A")
        axs[idx].set_title(f"Acc: {acc} | True: {true_label} | Pred: {predicted}", fontsize=8)
        
        # Format x-axis: Only show first few characters of partner IDs to avoid overcrowding
        if len(partner_counts) > 10:
            # If too many partners, only show every nth label
            n = max(1, len(partner_counts) // 10)
            axs[idx].set_xticks(range(0, len(partner_counts), n))
            
            # Highlight shared partners in x-tick labels with bold text
            xticklabels = []
            for i, p in enumerate(partner_counts.index[::n]):
                label = f"{p[:6]}..." 
                if p in shared_partners:
                    label = f"*{label}*"  # Add asterisk to mark shared partners
                xticklabels.append(label)
                
            axs[idx].set_xticklabels(xticklabels, rotation=45, fontsize=6)
        else:
            axs[idx].set_xticks(range(len(partner_counts)))
            
            # Highlight shared partners in x-tick labels
            xticklabels = []
            for p in partner_counts.index:
                label = f"{p[:6]}..." 
                if p in shared_partners:
                    label = f"*{label}*"  # Add asterisk to mark shared partners
                xticklabels.append(label)
                
            axs[idx].set_xticklabels(xticklabels, rotation=45, fontsize=6)
        
        # Add value labels above the bars
        for i, (p, v) in enumerate(zip(partner_counts.index, partner_counts.values)):
            if v > 0:  # Only add text for non-zero bars
                # Make shared partner counts bold and red
                if p in shared_partners:
                    axs[idx].text(i, v + 0.1, str(v), ha='center', fontsize=7, 
                                 color='red', fontweight='bold')
                else:
                    axs[idx].text(i, v + 0.1, str(v), ha='center', fontsize=6)
        
        axs[idx].set_ylabel('Transactions', fontsize=7)
        axs[idx].set_xlabel('External Partners', fontsize=7)
        
        # Add legend for transaction types and shared partners
        legend_elements = []
        
        # Add partner type colors to legend
        unique_types = set(partner_types.values())
        for t in unique_types:
            legend_elements.append(plt.Rectangle((0,0),1,1, color=colors.get(t, 'gray'), label=t))
        
        # Add shared partner indicator to legend
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='white', edgecolor='red', 
                                             linewidth=2, label='Shared Partner'))
        
        axs[idx].legend(handles=legend_elements, fontsize=6, loc='upper right')
    
    # Hide any unused subplots
    for idx in range(len(accounts_df), n_plots_square*n_plots_square):
        axs[idx].axis('off')
    
    # Add a global annotation about shared partners
    plt.figtext(0.5, 0.01, f"* Red borders indicate partners found in multiple accounts ({len(shared_partners)} partners)", 
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Make room for the annotation
    plt.savefig('plots_analyze/' + filename)
    
    # Return the shared partners for further analysis
    return shared_partners

@click.command()
# @click.argument('pred_csv', default="lvl_account/logs/2025.03.24_11.06.32_attn_cnn_val.csv", type=click.Path(exists=True))
@click.argument('pred_csv', type=click.Path(exists=True))
@click.argument('true_parquet', default='task/val_set/y_val.ver00.parquet', type=click.Path(exists=True))
@click.option('--transaction_file', type=click.Path(exists=True), default='task/val_set/x_val.ver08.parquet', help='CSV file with transaction data for in-depth analysis')
@click.option('--n_plots_square', type=int, default=4, help='Number of error accounts to plot')
@click.option('--full', is_flag=True, default=False, help='Perform full analysis with transaction plots')
def analyze_errors(pred_csv, true_parquet, transaction_file, n_plots_square, full):
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
        predicted_map = dict(zip(merged_df['AccountID'], merged_df['Fraudster_pred']))

        click.echo("Performing in-depth transaction plots: neighbor graphs")
        # Filter first n_errors error accounts for a grid and sort by true label (non fraudsters first)
        n_errors = min(n_plots_square**2, merged_df['error'].sum())
        error_accounts = merged_df[merged_df['error']].sample(n=n_errors, random_state=42)
        error_accounts = error_accounts.sort_values(by='Fraudster_true', ascending=True)
        plot_neighbor_graphs(error_accounts, predicted_map, merged_df, two_accounts_df, n_plots_square, "Neighbor Graphs for random wrong predictions", "neighbor_graph_subplots.pdf")
        click.echo("Neighbor graphs saved to neighbor_graph_subplots.pdf")

        # plot the average number of transactions per type for each account
        click.echo("Plotting transaction type distribution for error accounts...")
        plot_transaction_type_distribution(
            trans_df, 
            error_accounts, 
            merged_df, 
            n_plots_square, 
            "Transaction Type Distribution for Wrong Predictions", 
            "transaction_type_distribution_errors.pdf"
        )
        click.echo("Transaction type distribution saved to transaction_type_distribution_errors.pdf")

        # Add after the transaction type distribution plot:
        click.echo("Plotting external partner transaction frequency for error accounts...")
        plot_external_partner_frequency(
            trans_df, 
            error_accounts, 
            merged_df, 
            n_plots_square, 
            "External Partner Transaction Frequency for Wrong Predictions", 
            "external_partner_frequency_errors.pdf"
        )
        click.echo("External partner transaction frequency saved to external_partner_frequency_errors.pdf")

        if full:
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