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
    true_map = dict(zip(merged_df['AccountID'], merged_df['Fraudster_true']))
    graphs = []  # list to store each graph for an error account
    # Build graphs without drawing yet.
    for acc in error_accounts['AccountID']:
        G = nx.Graph()
        # For central node, assume external_type is 'customer' by default.
        G.add_node(acc)
        G.nodes[acc]['true'] = true_map.get(acc, 0)
        G.nodes[acc]['pred'] = predicted_map.get(acc, 0)
        G.nodes[acc]['external_type'] = 'customer'
        
        # Select neighbor transactions; assumes two_accounts_df has columns: AccountID, External, External_Type, Hour, Action, Amount
        neighbors = two_accounts_df[two_accounts_df['AccountID'] == acc]
        for _, row in neighbors.iterrows():
            nb = row['External']
            G.add_node(nb)
            G.nodes[nb]['true'] = true_map.get(nb, 0)
            G.nodes[nb]['pred'] = predicted_map.get(nb, 0)
            # Store external type from transaction (default to 'customer' if missing)
            G.nodes[nb]['external_type'] = row.get('External_Type', 'customer')
            edge_label = f"{row['Hour']}:{row['Action'][:1]}\n {row['Amount']}"
            G.add_edge(acc, nb, label=edge_label)
        graphs.append(G)
    
    # Compute global node count across all graphs.
    from collections import Counter
    global_node_count = Counter(n for G in graphs for n in G.nodes())
    
    # Create subplots and draw each graph
    fig, axs = plt.subplots(n_plots_square, n_plots_square, figsize=(n_plots_square * 3, n_plots_square * 3))
    axs = axs.flatten()
    
    for idx, G in enumerate(graphs):
        pos = nx.spring_layout(G)
        nx.draw_networkx_edges(G, pos, ax=axs[idx])
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=axs[idx])
        
        for n in G.nodes():
            x, y = pos[n]
            true_val = G.nodes[n].get('true', 0)
            pred_val = G.nodes[n].get('pred', 0)
            ext_type = G.nodes[n].get('external_type', 'customer')
            # Use det_color to set colors (red if fraudster, green if customer, blue if merchant, yellow if bank)
            fill_color = det_color(true_val, ext_type)
            border_color = det_color(pred_val, ext_type)
            axs[idx].scatter(x, y, s=600, facecolors=fill_color, edgecolors=border_color, linewidths=2, zorder=2)
            
            # Draw node label and highlight if this node appears more than once across graphs
            if global_node_count.get(n, 0) > 1:
                axs[idx].text(x, y, str(n), horizontalalignment="center", verticalalignment="center",
                              zorder=3, fontsize=10, fontweight="bold", color="blue")
            else:
                axs[idx].text(x, y, str(n), horizontalalignment="center", verticalalignment="center",
                              zorder=3, fontsize=8)
            # Annotate border with predicted value (offset downward)
            axs[idx].text(x, y - 0.1, f"P: {pred_val}", horizontalalignment="center", verticalalignment="center",
                          fontsize=7, color=border_color, zorder=4)
            
        axs[idx].set_title(f"Acc: {list(G.nodes())[0]} | Pred: {predicted_map.get(list(G.nodes())[0], 'N/A')}", fontsize=9)
        axs[idx].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('plots_analyze/' + filename)


def plot_reverse_neighbor_graphs(error_accounts, predicted_map, merged_df, trans_df, n_plots_square, title, filename):
    """
    Plot graphs where the central node is the error account, and neighbors are accounts 
    that have transactions with this account as their External partner.
    """
    true_map = dict(zip(merged_df['AccountID'], merged_df['Fraudster_true']))
    graphs = []
    for acc in error_accounts['AccountID']:
        G = nx.Graph()
        G.add_node(acc)
        G.nodes[acc]['true'] = true_map.get(acc, 0)
        G.nodes[acc]['pred'] = predicted_map.get(acc, 0)
        G.nodes[acc]['external_type'] = 'customer'
        
        # Find transactions where this account is the external partner.
        reverse_neighbors = trans_df[trans_df['External'] == acc]
        for _, row in reverse_neighbors.iterrows():
            nb = row['AccountID']
            G.add_node(nb)
            G.nodes[nb]['true'] = true_map.get(nb, 0)
            G.nodes[nb]['pred'] = predicted_map.get(nb, 0)
            G.nodes[nb]['external_type'] = row.get('External_Type', 'customer')
            edge_label = f"{row['Hour']}:{row['Action'][:1]}\n {row['Amount']}"
            G.add_edge(acc, nb, label=edge_label)
        graphs.append(G)
    
    # Compute global node count across all graphs.
    from collections import Counter
    global_node_count = Counter(n for G in graphs for n in G.nodes())
    
    fig, axs = plt.subplots(n_plots_square, n_plots_square, figsize=(n_plots_square * 3, n_plots_square * 3))
    axs = axs.flatten()
    
    for idx, G in enumerate(graphs):
        pos = nx.spring_layout(G)
        nx.draw_networkx_edges(G, pos, ax=axs[idx])
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=axs[idx])
        for n in G.nodes():
            x, y = pos[n]
            true_val = G.nodes[n].get('true', 0)
            pred_val = G.nodes[n].get('pred', 0)
            ext_type = G.nodes[n].get('external_type', 'customer')
            fill_color = det_color(true_val, ext_type)
            border_color = det_color(pred_val, ext_type)
            axs[idx].scatter(x, y, s=600, facecolors=fill_color, edgecolors=border_color, linewidths=2, zorder=2)
            if global_node_count.get(n, 0) > 1:
                axs[idx].text(x, y, str(n), horizontalalignment="center", verticalalignment="center",
                              zorder=3, fontsize=10, fontweight="bold", color="blue")
            else:
                axs[idx].text(x, y, str(n), horizontalalignment="center", verticalalignment="center",
                              zorder=3, fontsize=8)
            axs[idx].text(x, y - 0.1, f"P: {pred_val}", horizontalalignment="center", verticalalignment="center",
                          fontsize=7, color=border_color, zorder=4)
        axs[idx].set_title(f"Acc: {list(G.nodes())[0]} | Pred: {predicted_map.get(list(G.nodes())[0], 'N/A')}", fontsize=9)
        axs[idx].axis('off')
    
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
    Create histograms showing transaction frequencies with external partners for accounts.
    Highlight external partners that appear in multiple accounts.
    Also includes a summary histogram of partners by number of interactions (only for shared partners).
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
    
    # Create a figure with an additional subplot for the summary histogram
    fig = plt.figure(figsize=(n_plots_square*4, n_plots_square*4 + 3))
    
    # Create grid for individual account plots (top section)
    grid_size = n_plots_square*n_plots_square
    gs = fig.add_gridspec(n_plots_square+1, n_plots_square, height_ratios=[1]*n_plots_square + [0.8])
    
    # Create individual account axes
    axs = []
    for i in range(n_plots_square):
        for j in range(n_plots_square):
            ax = fig.add_subplot(gs[i, j])
            axs.append(ax)
    
    # Create summary histogram axis (bottom section)
    ax_summary = fig.add_subplot(gs[n_plots_square, :])
    
    # Plot individual account charts
    for idx, acc in enumerate(accounts_df['AccountID']):
        if idx >= len(axs):
            break
            
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
        
        # Format x-axis labels
        if len(partner_counts) > 10:
            n = max(1, len(partner_counts) // 10)
            axs[idx].set_xticks(range(0, len(partner_counts), n))
            
            xticklabels = []
            for i, p in enumerate(partner_counts.index[::n]):
                label = f"{p[:6]}..." 
                if p in shared_partners:
                    label = f"*{label}*"
                xticklabels.append(label)
                
            axs[idx].set_xticklabels(xticklabels, rotation=45, fontsize=6)
        else:
            axs[idx].set_xticks(range(len(partner_counts)))
            
            xticklabels = []
            for p in partner_counts.index:
                label = f"{p[:6]}..." 
                if p in shared_partners:
                    label = f"*{label}*"
                xticklabels.append(label)
                
            axs[idx].set_xticklabels(xticklabels, rotation=45, fontsize=6)
        
        # Add value labels above the bars
        for i, (p, v) in enumerate(zip(partner_counts.index, partner_counts.values)):
            if v > 0:
                if p in shared_partners:
                    axs[idx].text(i, v + 0.1, str(v), ha='center', fontsize=7, 
                                 color='red', fontweight='bold')
                else:
                    axs[idx].text(i, v + 0.1, str(v), ha='center', fontsize=6)
        
        axs[idx].set_ylabel('Transactions', fontsize=7)
        axs[idx].set_xlabel('External Partners', fontsize=7)
        
        # Add legend for transaction types and shared partners
        legend_elements = []
        
        unique_types = set(partner_types.values())
        for t in unique_types:
            legend_elements.append(plt.Rectangle((0,0),1,1, color=colors.get(t, 'gray'), label=t))
        
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='white', edgecolor='red', 
                                             linewidth=2, label='Shared Partner'))
        
        axs[idx].legend(handles=legend_elements, fontsize=6, loc='upper right')
    
    # Hide any unused subplots
    for idx in range(len(accounts_df), len(axs)):
        axs[idx].axis('off')
    
    # Replace the previous summary histogram with a bar chart of total interactions for shared partners.
    # Compute total interactions for each shared partner across all accounts.
    partner_interactions = {}
    for acc, acc_trans in account_transactions.items():
        if not acc_trans.empty:
            counts = acc_trans['External'].value_counts()
            for partner, cnt in counts.items():
                if partner in shared_partners:
                    partner_interactions[partner] = partner_interactions.get(partner, 0) + cnt
                    
    # Compute a global mapping for partner types for shared partners.
    partner_types_global = {}
    for acc, acc_trans in account_transactions.items():
        if not acc_trans.empty:
            acc_trans.loc[:, 'External_Type'] = acc_trans['External_Type'].fillna('Cash')
            for partner, group in acc_trans.groupby('External'):
                if partner in shared_partners:
                    ext_type = group['External_Type'].iloc[0]
                    partner_types_global.setdefault(partner, []).append(ext_type)
    
    # Determine dominant type for each shared partner and set colors accordingly.
    colors = {'Cash': 'lightgreen', 'bank': 'gold', 'merchant': 'skyblue', None: 'gray'}
    partner_colors = {}
    for p in partner_interactions.keys():
        types_list = partner_types_global.get(p, [])
        if types_list:
            dominant_type = max(set(types_list), key=types_list.count)
        else:
            dominant_type = 'Cash'
        partner_colors[p] = colors.get(dominant_type, 'gray')
    
    partners_sorted = sorted(partner_interactions.keys())
    interactions = [partner_interactions[p] for p in partners_sorted]
    
    # Plot the summary bar chart with color coding by external type.
    ax_summary = fig.add_subplot(gs[n_plots_square, :])
    bars = ax_summary.bar(
        range(len(partners_sorted)),
        interactions,
        color=[partner_colors[p] for p in partners_sorted],
        edgecolor='black'
    )
    
    ax_summary.set_xticks(range(len(partners_sorted)))
    ax_summary.set_xticklabels(partners_sorted, rotation=45, fontsize=8)
    ax_summary.set_xlabel('External Partner', fontsize=10)
    ax_summary.set_ylabel('Total Interactions', fontsize=10)
    ax_summary.set_title('Shared External Partners by Total Interactions', fontsize=12)
    
    for i, (p, v) in enumerate(zip(partners_sorted, interactions)):
        ax_summary.text(i, v + 0.1, str(v), ha='center', fontsize=9)
    
    ax_summary.grid(axis='y', linestyle='--', alpha=0.7)
    plt.figtext(0.5, 0.01, f"* Only external partners present in multiple accounts are shown (total: {len(shared_partners)})", 
                ha="center", fontsize=10, bbox={"facecolor":"orange","alpha":0.2, "pad":5})
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('plots_analyze/' + filename)
    
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

        # After the existing plot_neighbor_graphs call
        click.echo("Performing in-depth transaction plots: reverse neighbor graphs")
        plot_reverse_neighbor_graphs(error_accounts, predicted_map, merged_df, trans_df, n_plots_square, 
                                "Reverse Neighbor Graphs (accounts that transacted with these accounts)", 
                                "reverse_neighbor_graph_subplots.pdf")
        click.echo("Reverse neighbor graphs saved to reverse_neighbor_graph_subplots.pdf")

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

        # Select random non-fraudsters that were correctly predicted
        correct_nonfraud = merged_df[
            (merged_df['error'] == False) & 
            (merged_df['Fraudster_true'] == 0)
        ].sample(n=min(n_plots_square**2, sum((merged_df['error'] == False) & (merged_df['Fraudster_true'] == 0))), 
                random_state=42)

        # Generate comparison plot
        plot_external_partner_frequency(
            trans_df, 
            correct_nonfraud, 
            merged_df, 
            n_plots_square, 
            "External Partner Transaction Frequency for Correctly Predicted Non-Fraudsters", 
            "external_partner_frequency_nonfraud.pdf"
        )
        click.echo("External partner frequency comparison saved to external_partner_frequency_nonfraud.pdf")


        # After the existing plot_neighbor_graphs call
        click.echo("Performing in-depth transaction plots: reverse neighbor graphs")
        plot_reverse_neighbor_graphs(correct_nonfraud, predicted_map, merged_df, trans_df, n_plots_square, 
                                "Reverse Neighbor Graphs (accounts that transacted with these accounts)", 
                                "reverse_neighbor_graph_subplots_nonfraud.pdf")
        click.echo("Reverse neighbor graphs saved to reverse_neighbor_graph_subplots_nonfraud.pdf")


        click.echo(f"All missclassification AccountIDs: {error_accounts['AccountID'].to_list()}")
        click.echo(f"And their true label: {error_accounts['Fraudster_true'].to_list()}")
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
    # example usage:
    # python3 analyze_errors.py lvl_account/logs/2025.03.24_15.56.46_attn_cnn_val.csv --n_plots_square 8
    # python3 analyze_errors.py non_rescuable_errors.csv --n_plots_square 8
    # python3 analyze_errors.py lvl_account/logs/2025.03.31_11.46.13_simple_cnn_train.csv task/train_set/y_train.ver00.parquet  --n_plots_square 3 --full --transaction_file task/train_set/x_train.ver08.parquet
    analyze_errors()