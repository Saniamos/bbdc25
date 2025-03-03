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
    sns.countplot(data=merged_df, x="Action", hue="Fraudster", order=merged_df["Action"].value_counts().index)
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
    plt.title("Unique Account Counts")
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
    plt.title("Overdraft Counts")
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
    
    # Check if fraud data is provided before generating fraud-based plots
    if fraud_path:
        fraud_df = pd.read_csv(fraud_path)
        plot_fraud_action_difference(transactions_df, fraud_df, output_path=f"plot/{prefix}_fraud_action_difference.png")
        plot_fraud_percentage(transactions_df, fraud_df, output_path=f"plot/{prefix}_fraud_percentage.png")
        plot_summed_money_per_account(transactions_df, fraud_df, output_path=f"plot/{prefix}_summed_money_per_account.png")
        plot_transaction_traffic_per_account(transactions_df, fraud_df, output_path=f"plot/{prefix}_transaction_traffic_per_account.png")
    else:
        print("No fraud_path provided: Skipping fraud-based plots.")

if __name__ == "__main__":
    main()