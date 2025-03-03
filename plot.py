import click
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

@click.command()
@click.option('--transactions_path', default='/Users/yale/Repositories/bbdc25/task/train_set/x_train.csv', help='Path to the transactions CSV file')
@click.option('--fraud_path', default=None, help='Optional path to the fraud label CSV file')
@click.option('--val_name', default='x_val', help='Base filename for the output PNGs')
def main(transactions_path, fraud_path, val_name):
    transactions_df = pd.read_csv(transactions_path)
    prefix = f"{val_name}_"
    
    # Generate plots that only require transactions data
    plot_amount_distribution(transactions_df, output_path=f"plot/{prefix}_amount_distribution.png")
    plot_hourly_average_amount(transactions_df, output_path=f"plot/{prefix}_hourly_average_amount.png")
    plot_unique_accounts(transactions_df, output_path=f"plot/{prefix}_unique_accounts.png")
    plot_overdrafts(transactions_df, output_path=f"plot/{prefix}_overdrafts.png")
    plot_transfer_hops(transactions_df, output_path=f"plot/{prefix}_transfer_hops.png")
    
    # Check if fraud data is provided before generating fraud-based plots
    # Inside your main() function, after the other fraud-based plots
    if fraud_path:
        fraud_df = pd.read_csv(fraud_path)
        plot_fraud_action_difference(transactions_df, fraud_df, output_path=f"plot/{prefix}_fraud_action_difference.png")
        plot_fraud_percentage(transactions_df, fraud_df, output_path=f"plot/{prefix}_fraud_percentage.png")
        plot_summed_money_per_account(transactions_df, fraud_df, output_path=f"plot/{prefix}_summed_money_per_account.png")
        plot_transaction_traffic_per_account(transactions_df, fraud_df, output_path=f"plot/{prefix}_transaction_traffic_per_account.png")
        plot_fraud_overdrafts(transactions_df, fraud_df, output_path=f"plot/{prefix}_fraud_overdrafts.png")
        plot_unique_accounts_by_fraud(fraud_df, output_path=f"plot/{prefix}_unique_accounts_by_fraud.png")
        # New plots:
        plot_transfer_counts_per_account(transactions_df, fraud_df, output_path=f"plot/{prefix}_transfer_counts_per_account.png")
        plot_fraudsters_by_external_type(transactions_df, fraud_df, output_path=f"plot/{prefix}_fraudsters_by_external_type.png")
    else:
        print("No fraud_path provided: Skipping fraud-based plots.")

if __name__ == "__main__":
    main()