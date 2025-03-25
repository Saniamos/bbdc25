import click
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Use seaborn style
sns.set(style="whitegrid")

def plot_amount_distribution(transactions_df, output_path):
    plt.figure(figsize=(8, 6))
    sns.histplot(transactions_df["Amount"], bins=30, kde=True)
    plt.title("Distribution of Transaction Amounts")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved amount distribution plot to {output_path}")

    output_path = output_path.replace('.png', '_log.png')
    plt.figure(figsize=(8, 6))
    sns.histplot(np.log(transactions_df["Amount"]), bins=30, kde=True)
    plt.title("Distribution of Transaction Amounts")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved amount distribution plot to {output_path}")

def plot_hourly_average_amount(transactions_df, output_path):
    hourly_avg = transactions_df.groupby("Hour")["Amount"].mean().reset_index()
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=hourly_avg, x="Hour", y="Amount", marker="o")
    plt.title("Hourly Average Transaction Amount")
    plt.xlabel("Hour")
    plt.ylabel("Average Amount")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved hourly average amount plot to {output_path}")

def plot_fraud_action_difference(transactions_df, fraud_df, output_path):
    transactions_df["AccountID"] = transactions_df["AccountID"].astype(str)
    fraud_df["AccountID"] = fraud_df["AccountID"].astype(str)
    merged_df = pd.merge(transactions_df, fraud_df, on="AccountID", how="inner")
    plt.figure(figsize=(10, 6))
    sns.countplot(data=merged_df, x="Action", hue="Fraudster", 
                  order=merged_df["Action"].value_counts().index)
    plt.title("Transaction Action Counts by Fraudulent Status")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.legend(title="Fraudster", labels=["Non-fraud", "Fraud"])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved fraud action difference plot to {output_path}")

def plot_fraud_percentage(transactions_df, fraud_df, output_path):
    """
    For each transaction action, calculate the percentage of transactions performed
    by fraud vs non-fraud accounts and display as a stacked bar chart with annotations.
    """
    transactions_df["AccountID"] = transactions_df["AccountID"].astype(str)
    fraud_df["AccountID"] = fraud_df["AccountID"].astype(str)
    merged = pd.merge(transactions_df, fraud_df, on="AccountID", how="inner")
    counts = merged.groupby(["Action", "Fraudster"]).size().unstack(fill_value=0)
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(10, 6))
    ax = percentages.plot(kind="bar", stacked=True, figsize=(10, 6),
                          color=[sns.color_palette("pastel")[0], sns.color_palette("pastel")[1]])
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            x = p.get_x() + p.get_width() / 2
            y = p.get_y() + height / 2
            ax.text(x, y, f'{height:.1f}%', ha='center', va='center', color="black", fontsize=9)
    plt.title("Fraudulent vs Non-Fraudulent Percentage by Action")
    plt.xlabel("Action")
    plt.ylabel("Percentage")
    plt.legend(title="Fraudster", labels=["Non-fraud", "Fraud"])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved fraud percentage plot to {output_path}")

def plot_unique_accounts(transactions_df, output_path):
    internal_count = transactions_df["AccountID"].nunique()
    external_count = transactions_df["External"].dropna().nunique()
    plt.figure(figsize=(6, 6))
    sns.barplot(x=["Internal", "External"], y=[internal_count, external_count], palette="viridis")
    plt.title("Unique Account Counts (All Transactions)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved unique accounts plot to {output_path}")

def plot_overdrafts(transactions_df, output_path):
    counts = transactions_df["isUnauthorizedOverdraft"].value_counts()
    labels = ["No Overdraft", "Overdraft"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    plt.figure(figsize=(6, 6))
    sns.barplot(x=labels, y=values, palette="magma")
    plt.title("Overdraft Counts (All Transactions)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved overdrafts plot to {output_path}")

def plot_summed_money_per_account(transactions_df, fraud_df, output_path):
    sum_money = transactions_df.groupby("AccountID")["Amount"].sum().reset_index()
    fraud_df["AccountID"] = fraud_df["AccountID"].astype(str)
    sum_money["AccountID"] = sum_money["AccountID"].astype(str)
    merged = pd.merge(sum_money, fraud_df, on="AccountID", how="inner")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=merged, x="Fraudster", y="Amount", palette="Set2")
    plt.title("Summed Money per Account by Fraud Status")
    plt.xlabel("Fraudster (0 = Non-fraud, 1 = Fraud)")
    plt.ylabel("Total Amount")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved summed money per account plot to {output_path}")

def plot_transaction_traffic_per_account(transactions_df, fraud_df, output_path):
    traffic = transactions_df.groupby("AccountID").size().reset_index(name="TransactionCount")
    fraud_df["AccountID"] = fraud_df["AccountID"].astype(str)
    traffic["AccountID"] = traffic["AccountID"].astype(str)
    merged = pd.merge(traffic, fraud_df, on="AccountID", how="inner")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=merged, x="Fraudster", y="TransactionCount", palette="Set3")
    plt.title("Transaction Traffic per Account by Fraud Status")
    plt.xlabel("Fraudster (0 = Non-fraud, 1 = Fraud)")
    plt.ylabel("Transaction Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved transaction traffic plot to {output_path}")

def plot_fraud_overdrafts(transactions_df, fraud_df, output_path):
    """
    Plot the overdraft counts among fraudsters.
    """
    transactions_df["AccountID"] = transactions_df["AccountID"].astype(str)
    fraud_df["AccountID"] = fraud_df["AccountID"].astype(str)
    merged = pd.merge(transactions_df, fraud_df, on="AccountID", how="inner")
    fraudsters = merged[merged["Fraudster"] == 1]
    counts = fraudsters["isUnauthorizedOverdraft"].value_counts()
    labels = ["No Overdraft", "Overdraft"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    plt.figure(figsize=(6, 6))
    sns.barplot(x=labels, y=values, palette="coolwarm")
    plt.title("Overdraft Occurrence Among Fraudsters")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved fraud overdrafts plot to {output_path}")

def plot_unique_accounts_by_fraud(fraud_df, output_path):
    """
    Distinguish unique accounts into fraud vs non-fraud using fraud label data.
    """
    counts = fraud_df["Fraudster"].value_counts().sort_index()
    labels = ["Non-fraud", "Fraud"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    plt.figure(figsize=(6, 6))
    sns.barplot(x=labels, y=values, palette="pastel")
    plt.title("Unique Accounts by Fraud Status")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved unique accounts by fraud plot to {output_path}")

def plot_transfer_hops(transactions_df, output_path):
    """
    For transfer transactions (Action=="TRANSFER"), group transfers into chains
    where the recipient of one transfer (External) is the sender (AccountID) in the next.
    The plot shows a scatter plot with the number of transfers (hops) in the chain versus the total amount transferred.
    """
    # Filter for transfers and sort by Hour (assuming this proxies the order)
    transfers = transactions_df[transactions_df["Action"]=="TRANSFER"].copy()
    transfers.sort_values("Hour", inplace=True)
    
    chains = []
    current_chain = []
    for _, row in transfers.iterrows():
        if not current_chain:
            current_chain.append(row)
        else:
            last = current_chain[-1]
            # If this transfer's sender equals the previous transfer's recipient, extend the chain
            if row["AccountID"] == last["External"]:
                current_chain.append(row)
            else:
                chains.append(current_chain)
                current_chain = [row]
    if current_chain:
        chains.append(current_chain)
    
    chain_lengths = []
    chain_amounts = []
    for chain in chains:
        chain_lengths.append(len(chain))  # number of hops (transfers)
        chain_amounts.append(sum(tx["Amount"] for _, tx in pd.DataFrame(chain).iterrows()))
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=chain_lengths, y=chain_amounts)
    plt.xlabel("Number of Hops (Transfers in Chain)")
    plt.ylabel("Total Amount Transferred")
    plt.title("Transfer Chains: Hops vs Total Amount Transferred")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved transfer hops plot to {output_path}")

def plot_transfer_counts_per_account(transactions_df, fraud_df, output_path):
    """
    Count the number of transfer transactions per account (Action=="TRANSFER")
    and plot a box plot of transfer counts grouped by fraud status.
    """
    # Filter transfer transactions
    transfers_df = transactions_df[transactions_df["Action"]=="TRANSFER"].copy()
    # Group transfers per AccountID
    counts = transfers_df.groupby("AccountID").size().reset_index(name="TransferCount")
    # Ensure AccountID is a string and merge with fraud labels
    counts["AccountID"] = counts["AccountID"].astype(str)
    fraud_df["AccountID"] = fraud_df["AccountID"].astype(str)
    merged = pd.merge(counts, fraud_df, on="AccountID", how="inner")
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=merged, x="Fraudster", y="TransferCount", palette="Set3")
    plt.xlabel("Fraudster (0 = Non-fraud, 1 = Fraud)")
    plt.ylabel("Number of Transfers")
    plt.title("Transfers per Account by Fraud Status")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved transfer counts per account plot to {output_path}")

def plot_fraudsters_by_external_type(transactions_df, fraud_df, output_path):
    """
    Count fraudster transactions by external account type.
    The external type is determined by the first character of the 'External' field:
      - 'C' -> "customer"
      - 'B' -> "bank"
      - 'M' -> "merchant"
      - otherwise "other"
    """
    def get_external_type(x):
        if pd.isna(x):
            return "unknown"
        code = str(x)[0].lower()
        if code == 'b':
            return "bank"
        elif code == 'm':
            return "merchant"
        elif code == 'c':
            return "customer"
        else:
            return "other"
    
    # Ensure AccountID are strings and merge transactions with fraud labels
    transactions_df["AccountID"] = transactions_df["AccountID"].astype(str)
    fraud_df["AccountID"] = fraud_df["AccountID"].astype(str)
    merged = pd.merge(transactions_df, fraud_df, on="AccountID", how="inner")
    # Filter for fraudsters only
    fraudsters = merged[merged["Fraudster"] == 1].copy()
    # Create External_Type if not already present
    if "External_Type" not in fraudsters.columns:
        fraudsters["External_Type"] = fraudsters["External"].apply(get_external_type)
    
    counts = fraudsters["External_Type"].value_counts().reset_index()
    counts.columns = ["External_Type", "Count"]
    
    plt.figure(figsize=(6, 6))
    sns.barplot(data=counts, x="External_Type", y="Count", palette="pastel")
    plt.title("Fraudster Count by External Account Type")
    plt.xlabel("External Account Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved fraudsters by external type plot to {output_path}")

def plot_fraud_percentage_pie(fraud_df, output_path):
    """
    Create a pie chart showing the percentage breakdown of fraudsters vs non-fraudsters.
    Includes actual numbers in the chart labels.
    """
    # Count fraudsters and non-fraudsters
    counts = fraud_df["Fraudster"].value_counts().sort_index()
    non_fraud_count = counts.get(0, 0)
    fraud_count = counts.get(1, 0)
    total = non_fraud_count + fraud_count
    
    # Calculate percentages
    labels = [f'Non-fraud\n{non_fraud_count} ({non_fraud_count/total:.1%})', 
              f'Fraud\n{fraud_count} ({fraud_count/total:.1%})']
    sizes = [non_fraud_count, fraud_count]
    
    # Create pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='', startangle=90,
            colors=[sns.color_palette("pastel")[0], sns.color_palette("pastel")[1]],
            wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
    plt.title("Fraudster Distribution in Dataset")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved fraud percentage pie chart to {output_path}")

def plot_transactions_per_account(transactions_df, output_path):
    """
    Create a histogram showing the distribution of the number of transactions per account.
    """
    # Count transactions per account
    tx_counts = transactions_df.groupby("AccountID").size()
    
    # Find account with most transactions
    max_tx_account = tx_counts.idxmax()
    max_tx_count = tx_counts.max()
    print(f"Account with most transactions: {max_tx_account} (Count: {max_tx_count})")
    
    plt.figure(figsize=(8, 6))
    sns.histplot(tx_counts, bins=30, kde=True)
    plt.title("Distribution of Transaction Counts per Account")
    plt.xlabel("Number of Transactions")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved transactions per account histogram to {output_path}")

def plot_cash_only_accounts(transactions_df, fraud_df, output_path):
    """
    Identify accounts that only perform CASH_IN and CASH_OUT transactions,
    and analyze how many of those are fraudulent with detailed breakdowns.
    
    Creates 4 pie charts:
    1. Cash-only vs other accounts (all accounts)
    2. Cash-in-only vs cash-out-only vs both (for cash-only accounts)
    3. Fraud status of cash-only accounts
    4. Fraud percentage by cash transaction type
    
    Args:
        transactions_df: DataFrame with transaction data
        fraud_df: DataFrame with fraud labels
        output_path: Path to save the output plot
    """
    # Ensure AccountIDs are strings
    transactions_df["AccountID"] = transactions_df["AccountID"].astype(str)
    fraud_df["AccountID"] = fraud_df["AccountID"].astype(str)
    
    # Group transactions by account and get unique action types for each
    account_actions = transactions_df.groupby("AccountID")["Action"].unique().reset_index()
    
    # Identify accounts that only have CASH_IN and CASH_OUT transactions
    cash_only_accounts = account_actions[account_actions["Action"].apply(
        lambda x: set(x).issubset({"CASH_IN", "CASH_OUT"})
    )]
    
    # Further categorize cash-only accounts
    cash_only_accounts["Cash_Type"] = cash_only_accounts["Action"].apply(
        lambda x: "Cash-in only" if set(x) == {"CASH_IN"} else 
                  "Cash-out only" if set(x) == {"CASH_OUT"} else 
                  "Both cash-in & cash-out"
    )
    
    # Count total accounts
    total_accounts = transactions_df["AccountID"].nunique()
    cash_only_count = len(cash_only_accounts)
    other_accounts_count = total_accounts - cash_only_count
    
    # Get cash type counts
    cash_type_counts = cash_only_accounts["Cash_Type"].value_counts()
    
    # Join with fraud data to determine fraud stats
    merged = pd.merge(cash_only_accounts, fraud_df, on="AccountID", how="inner")
    fraud_counts = merged["Fraudster"].value_counts().sort_index()
    non_fraud_count = fraud_counts.get(0, 0)
    fraud_count = fraud_counts.get(1, 0)
    
    # Get fraud stats by cash type
    fraud_by_type = merged.groupby("Cash_Type")["Fraudster"].value_counts().unstack(fill_value=0)
    if 0 not in fraud_by_type.columns:
        fraud_by_type[0] = 0
    if 1 not in fraud_by_type.columns:
        fraud_by_type[1] = 0
    
    # Setup subplots for 4 pie charts
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Cash-only vs other accounts (all accounts) - Top Left
    labels1 = [f"Cash-only\n{cash_only_count} ({cash_only_count/total_accounts:.1%})",
               f"Other accounts\n{other_accounts_count} ({other_accounts_count/total_accounts:.1%})"]
    axes[0, 0].pie([cash_only_count, other_accounts_count], labels=labels1, autopct='',
                  colors=sns.color_palette("pastel"), wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    axes[0, 0].set_title("All Accounts: Cash-Only vs Other Accounts", fontsize=12)
    
    # 2. Cash-in-only vs cash-out-only vs both - Top Right
    cash_in_only = cash_type_counts.get("Cash-in only", 0)
    cash_out_only = cash_type_counts.get("Cash-out only", 0)
    both = cash_type_counts.get("Both cash-in & cash-out", 0)
    
    labels2 = [f"Cash-in only\n{cash_in_only} ({cash_in_only/cash_only_count:.1%})",
               f"Cash-out only\n{cash_out_only} ({cash_out_only/cash_only_count:.1%})",
               f"Both\n{both} ({both/cash_only_count:.1%})"]
    
    axes[0, 1].pie([cash_in_only, cash_out_only, both], labels=labels2, autopct='',
                  colors=sns.color_palette("pastel", 3), wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    axes[0, 1].set_title("Cash-Only Accounts by Transaction Type", fontsize=12)
    
    # 3. Fraud status of cash-only accounts - Bottom Left
    if non_fraud_count + fraud_count > 0:
        labels3 = [f"Non-fraudulent\n{non_fraud_count} ({non_fraud_count/(non_fraud_count+fraud_count):.1%})",
                   f"Fraudulent\n{fraud_count} ({fraud_count/(non_fraud_count+fraud_count):.1%})"]
        
        axes[1, 0].pie([non_fraud_count, fraud_count], labels=labels3, autopct='',
                      colors=[sns.color_palette("pastel")[0], sns.color_palette("pastel")[1]],
                      wedgeprops={'edgecolor': 'black', 'linewidth': 1})
        axes[1, 0].set_title("Cash-Only Accounts: Fraud Status", fontsize=12)
    else:
        axes[1, 0].text(0.5, 0.5, "No cash-only accounts found in fraud dataset",
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # 4. Fraud percentage by cash transaction type - Bottom Right
    if len(fraud_by_type) > 0:
        # Prepare data for the stacked bar chart as pie charts
        types = fraud_by_type.index.tolist()
        
        # If we have multiple cash types with fraud data
        if len(types) > 1:
            # Create a nested pie chart
            fraud_pcts = []
            for cash_type in types:
                if cash_type in fraud_by_type.index:
                    non_fraud = fraud_by_type.loc[cash_type, 0]
                    fraud = fraud_by_type.loc[cash_type, 1]
                    total = non_fraud + fraud
                    if total > 0:
                        fraud_pct = fraud / total
                    else:
                        fraud_pct = 0
                    fraud_pcts.append((cash_type, fraud_pct, total))
            
            # Sort by fraud percentage for the plot
            fraud_pcts.sort(key=lambda x: x[1], reverse=True)
            
            # Create labels and sizes for the plot
            labels4 = [f"{cash_type}\n{total} accounts ({fraud_pct:.1%} fraud)" 
                      for cash_type, fraud_pct, total in fraud_pcts]
            sizes4 = [total for _, _, total in fraud_pcts]
            
            axes[1, 1].pie(sizes4, labels=labels4, autopct='',
                          colors=sns.color_palette("pastel", len(fraud_pcts)), 
                          wedgeprops={'edgecolor': 'black', 'linewidth': 1})
            axes[1, 1].set_title("Cash-Only Accounts by Type with Fraud Percentage", fontsize=12)
        else:
            # Only one cash type, show fraud vs non-fraud for that type
            cash_type = types[0]
            non_fraud = fraud_by_type.loc[cash_type, 0]
            fraud = fraud_by_type.loc[cash_type, 1]
            
            labels4 = [f"Non-fraudulent\n{non_fraud} ({non_fraud/(non_fraud+fraud):.1%})",
                      f"Fraudulent\n{fraud} ({fraud/(non_fraud+fraud):.1%})"]
            
            axes[1, 1].pie([non_fraud, fraud], labels=labels4, autopct='',
                          colors=[sns.color_palette("pastel")[0], sns.color_palette("pastel")[1]], 
                          wedgeprops={'edgecolor': 'black', 'linewidth': 1})
            axes[1, 1].set_title(f"Fraud Status for {cash_type} Accounts", fontsize=12)
    else:
        axes[1, 1].text(0.5, 0.5, "No cash-only accounts found with fraud data",
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    # Adjust layout and save
    plt.suptitle("Analysis of Accounts That Only Perform Cash Transactions", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    
    # Print detailed statistics
    print(f"\nCash-only accounts: {cash_only_count} out of {total_accounts} ({cash_only_count/total_accounts:.1%})")
    print(f"  - Cash-in only: {cash_in_only} ({cash_in_only/cash_only_count:.1%})")
    print(f"  - Cash-out only: {cash_out_only} ({cash_out_only/cash_only_count:.1%})")
    print(f"  - Both cash-in & cash-out: {both} ({both/cash_only_count:.1%})")
    
    if non_fraud_count + fraud_count > 0:
        print(f"\nFraudulent cash-only accounts: {fraud_count} out of {non_fraud_count + fraud_count} cash-only accounts ({fraud_count/(non_fraud_count+fraud_count):.1%})")
        
        for cash_type in types:
            if cash_type in fraud_by_type.index:
                non_fraud = fraud_by_type.loc[cash_type, 0]
                fraud = fraud_by_type.loc[cash_type, 1]
                total = non_fraud + fraud
                if total > 0:
                    print(f"  - {cash_type}: {fraud} fraudulent out of {total} ({fraud/total:.1%})")
    
    print(f"\nSaved cash-only accounts analysis to {output_path}")

def plot_accounts_by_transaction_count(transactions_df, fraud_df, output_path, max_tx=1000, bin_size=3):
    """
    Create a stacked bar chart showing the distribution of accounts by number of transactions,
    with bars stacked by fraud/non-fraud status.
    
    Args:
        transactions_df: DataFrame with transaction data
        fraud_df: DataFrame with fraud labels
        output_path: Path to save the output plot
        max_tx: Maximum number of transactions to show on x-axis (higher counts grouped)
        bin_size: Size of each bin/group of transaction counts
    """
    # Ensure AccountIDs are strings
    transactions_df["AccountID"] = transactions_df["AccountID"].astype(str)
    fraud_df["AccountID"] = fraud_df["AccountID"].astype(str)
    
    # Count transactions per account
    tx_counts = transactions_df.groupby("AccountID").size().reset_index(name="TransactionCount")
    
    # Merge with fraud data
    merged = pd.merge(tx_counts, fraud_df, on="AccountID", how="inner")
    
    # Create transaction count bins
    def bin_transactions(count):
        if count > max_tx:
            return f"{max_tx}+"
        lower = (count // bin_size) * bin_size
        upper = lower + bin_size - 1
        return f"{lower}-{upper}"
    
    merged["TxCountBin"] = merged["TransactionCount"].apply(bin_transactions)
    
    # Group by bin and fraud status to count accounts
    accounts_by_bin = merged.groupby(["TxCountBin", "Fraudster"]).size().unstack(fill_value=0)
    
    # Sort bins properly
    def sort_key(bin_label):
        if bin_label.endswith("+"):
            return float('inf')  # Put the "max_tx+" bin at the end
        return int(bin_label.split("-")[0])
    
    accounts_by_bin = accounts_by_bin.reindex(sorted(accounts_by_bin.index, key=sort_key))
    
    # Rename columns for clarity
    if 0 in accounts_by_bin.columns:
        accounts_by_bin.rename(columns={0: "Non-fraud"}, inplace=True)
    else:
        accounts_by_bin["Non-fraud"] = 0
        
    if 1 in accounts_by_bin.columns:
        accounts_by_bin.rename(columns={1: "Fraud"}, inplace=True)
    else:
        accounts_by_bin["Fraud"] = 0
    
    # Create stacked bar chart
    plt.figure(figsize=(40, 7))
    accounts_by_bin.plot(kind="bar", stacked=True, figsize=(12, 7), 
                         color=[sns.color_palette("pastel")[0], sns.color_palette("pastel")[1]])
    
    # Add value labels on bars
    ax = plt.gca()
    for c in ax.containers:
        # Add value labels
        labels = [f'{int(v)}' if v > 0 else '' for v in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='center', rotation=90)
    
    # Calculate and add fraud percentage to each bin
    for i, (idx, row) in enumerate(accounts_by_bin.iterrows()):
        total = row["Non-fraud"] + row["Fraud"]
        if total > 0:
            fraud_pct = row["Fraud"] / total * 100
            if fraud_pct > 0:
                plt.text(i, row.sum() + 5, f"{fraud_pct:.1f}% fraud", 
                         ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.title("Account Distribution by Transaction Count and Fraud Status", fontsize=14)
    plt.xlabel("Number of Transactions per Account", fontsize=12)
    plt.ylabel("Number of Accounts", fontsize=12)
    plt.legend(title="Account Type")
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved account distribution by transaction count plot to {output_path}")

def plot_accounts_by_external_count(transactions_df, fraud_df, output_path, max_ext=30, bin_size=1):
    """
    Create a stacked bar chart showing the distribution of accounts by number of distinct external IDs
    they interact with, with bars stacked by fraud/non-fraud status.
    
    Args:
        transactions_df: DataFrame with transaction data
        fraud_df: DataFrame with fraud labels
        output_path: Path to save the output plot
        max_ext: Maximum number of external IDs to show on x-axis (higher counts grouped)
        bin_size: Size of each bin/group of external ID counts
    """
    # Ensure AccountIDs are strings
    transactions_df["AccountID"] = transactions_df["AccountID"].astype(str)
    fraud_df["AccountID"] = fraud_df["AccountID"].astype(str)
    
    # Drop rows where External is NaN (no external account involved)
    ext_df = transactions_df.dropna(subset=["External"])
    
    # Count distinct external IDs per account
    ext_counts = ext_df.groupby("AccountID")["External"].nunique().reset_index(name="ExternalCount")
    
    # Merge with fraud data
    merged = pd.merge(ext_counts, fraud_df, on="AccountID", how="inner")
    
    # Create external count bins
    def bin_externals(count):
        if count > max_ext:
            return f"{max_ext}+"
        lower = (count // bin_size) * bin_size
        upper = lower + bin_size - 1
        if lower == upper:
            return f"{lower}"
        return f"{lower}-{upper}"
    
    merged["ExtCountBin"] = merged["ExternalCount"].apply(bin_externals)
    
    # Group by bin and fraud status to count accounts
    accounts_by_bin = merged.groupby(["ExtCountBin", "Fraudster"]).size().unstack(fill_value=0)
    
    # Sort bins properly
    def sort_key(bin_label):
        if bin_label.endswith("+"):
            return float('inf')  # Put the "max_ext+" bin at the end
        if "-" in bin_label:
            return int(bin_label.split("-")[0])
        return int(bin_label)
    
    accounts_by_bin = accounts_by_bin.reindex(sorted(accounts_by_bin.index, key=sort_key))
    
    # Rename columns for clarity
    if 0 in accounts_by_bin.columns:
        accounts_by_bin.rename(columns={0: "Non-fraud"}, inplace=True)
    else:
        accounts_by_bin["Non-fraud"] = 0
        
    if 1 in accounts_by_bin.columns:
        accounts_by_bin.rename(columns={1: "Fraud"}, inplace=True)
    else:
        accounts_by_bin["Fraud"] = 0
    
    # Create stacked bar chart
    plt.figure(figsize=(12, 7))
    accounts_by_bin.plot(kind="bar", stacked=True, figsize=(12, 7), 
                         color=[sns.color_palette("pastel")[0], sns.color_palette("pastel")[1]])
    
    # Add value labels on bars
    ax = plt.gca()
    for c in ax.containers:
        # Add value labels
        labels = [f'{int(v)}' if v > 0 else '' for v in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='center')
    
    # Calculate and add fraud percentage to each bin
    for i, (idx, row) in enumerate(accounts_by_bin.iterrows()):
        total = row["Non-fraud"] + row["Fraud"]
        if total > 0:
            fraud_pct = row["Fraud"] / total * 100
            if fraud_pct > 0:
                plt.text(i, row.sum() + 5, f"{fraud_pct:.1f}% fraud", 
                         ha='center', va='bottom', fontsize=8)
    
    plt.title("Account Distribution by Distinct External Accounts and Fraud Status", fontsize=14)
    plt.xlabel("Number of Distinct External Accounts per Account", fontsize=12)
    plt.ylabel("Number of Accounts", fontsize=12)
    plt.legend(title="Account Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved account distribution by external count plot to {output_path}")


@click.command()
@click.option('--transactions_path', default='/Users/yale/Repositories/bbdc25/task/train_set/x_train.csv', help='Path to the transactions CSV file')
@click.option('--fraud_path', default=None, help='Optional path to the fraud label CSV file')
@click.option('--val_name', default='x_val', help='Base filename for the output PNGs')
def main(transactions_path, fraud_path, val_name):
    transactions_df = pd.read_parquet(transactions_path)
    prefix = f"{val_name}_"
    
    # Generate plots that only require transactions data
    plot_amount_distribution(transactions_df, output_path=f"plot/{prefix}_amount_distribution.png")
    # plot_hourly_average_amount(transactions_df, output_path=f"plot/{prefix}_hourly_average_amount.png")
    # plot_unique_accounts(transactions_df, output_path=f"plot/{prefix}_unique_accounts.png")
    # plot_overdrafts(transactions_df, output_path=f"plot/{prefix}_overdrafts.png")
    # plot_transfer_hops(transactions_df, output_path=f"plot/{prefix}_transfer_hops.png")
    # plot_transactions_per_account(transactions_df, output_path=f"plot/{prefix}_transactions_per_account.png")

    # Check if fraud data is provided before generating fraud-based plots
    # Inside your main() function, after the other fraud-based plots
    if fraud_path:
        fraud_df = pd.read_parquet(fraud_path)
        # plot_fraud_percentage_pie(fraud_df, output_path=f"plot/{prefix}_fraud_percentage_pie.png")
        # plot_fraud_action_difference(transactions_df, fraud_df, output_path=f"plot/{prefix}_fraud_action_difference.png")
        # plot_fraud_percentage(transactions_df, fraud_df, output_path=f"plot/{prefix}_fraud_percentage.png")
        # plot_summed_money_per_account(transactions_df, fraud_df, output_path=f"plot/{prefix}_summed_money_per_account.png")
        # plot_transaction_traffic_per_account(transactions_df, fraud_df, output_path=f"plot/{prefix}_transaction_traffic_per_account.png")
        # plot_fraud_overdrafts(transactions_df, fraud_df, output_path=f"plot/{prefix}_fraud_overdrafts.png")
        # plot_unique_accounts_by_fraud(fraud_df, output_path=f"plot/{prefix}_unique_accounts_by_fraud.png")
        # plot_transfer_counts_per_account(transactions_df, fraud_df, output_path=f"plot/{prefix}_transfer_counts_per_account.png")
        # plot_fraudsters_by_external_type(transactions_df, fraud_df, output_path=f"plot/{prefix}_fraudsters_by_external_type.png")
        # plot_cash_only_accounts(transactions_df, fraud_df, output_path=f"plot/{prefix}_cash_only_accounts.png")
        # plot_accounts_by_transaction_count(transactions_df, fraud_df, output_path=f"plot/{prefix}_accounts_by_tx_count.png")
        # plot_accounts_by_external_count(transactions_df, fraud_df, output_path=f"plot/{prefix}_accounts_by_external_count.png", max_ext=20, bin_size=1)
        pass
    else:
        print("No fraud_path provided: Skipping fraud-based plots.")

if __name__ == "__main__":
    main()