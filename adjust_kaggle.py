import click
import pandas as pd
import os

@click.command()
@click.option('--input_path', required=True, help='Path to the input Kaggle CSV file')
@click.option('--output_path', required=True, help='Path to write the adjusted CSV file')
@click.option('--fraud_output', required=False, help='Path to write the fraud label CSV file (AccountID,Fraudster).')
def adjust_kaggle(input_path, output_path, fraud_output):
    """
    Transforms a Kaggle CSV file with columns:
      step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud
    into a CSV with columns:
      Hour,Action,Amount,AccountID,External,OldBalance,NewBalance,isUnauthorizedOverdraft,External_Type
    
    Additionally, if --fraud_output is provided, a second CSV is created with columns:
      AccountID,Fraudster
    where Fraudster is computed as the maximum value of isFraud per account (i.e. an account is marked fraud if any transaction is fraud).
    
    Transformation rules:
      - Hour = step
      - Action = "CASH_IN" if type is PAYMENT; else the original 'type'
      - Amount = amount
      - AccountID = nameOrig
      - External = "" if type is PAYMENT; if TRANSFER, then nameDest
      - OldBalance = oldbalanceOrg
      - NewBalance = newbalanceOrig
      - isUnauthorizedOverdraft = 0 (default)
      - External_Type = computed from External (using first letter mapping: c -> customer, b -> bank, m -> merchant)
    """
    # Read the input CSV
    df = pd.read_csv(input_path)
    
    # Prepare new DataFrame with desired columns.
    new_df = pd.DataFrame()
    
    # Hour
    new_df["Hour"] = df["step"]
    
    # Action: Map PAYMENT to CASH_IN, keep other types as their original value.
    new_df["Action"] = df["type"].apply(lambda x: "CASH_IN" if x.upper() == "PAYMENT" else x.upper())
    
    # Amount remains the same.
    new_df["Amount"] = df["amount"]
    
    # AccountID from nameOrig.
    new_df["AccountID"] = df["nameOrig"]
    
    # External: if the transaction is PAYMENT, leave External blank; else use nameDest.
    new_df["External"] = df.apply(lambda row: "" if row["type"].upper() == "PAYMENT" else row["nameDest"], axis=1)
    
    # OldBalance and NewBalance: use the original organization balances.
    new_df["OldBalance"] = df["oldbalanceOrg"]
    new_df["NewBalance"] = df["newbalanceOrig"]
    
    # isUnauthorizedOverdraft is set to 0.
    new_df["isUnauthorizedOverdraft"] = 0
    
    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_df.to_csv(output_path, index=False)
    print(f"Adjusted file written to {output_path}")
    
    # If fraud_output is provided, create the fraud label file.
    if fraud_output:
        # Group by nameOrig and take the maximum of isFraud so that if any transaction is fraudulent, account is marked fraud.
        fraud_df = df.groupby("nameOrig", as_index=False)["isFraud"].max()
        fraud_df.rename(columns={"nameOrig": "AccountID", "isFraud": "Fraudster"}, inplace=True)
        os.makedirs(os.path.dirname(fraud_output), exist_ok=True)
        fraud_df.to_csv(fraud_output, index=False)
        print(f"Fraud label file written to {fraud_output}")

if __name__ == "__main__":
    # Example usage:
    # python adjust_kaggle.py --input_path task_orig/kaggle_set/kaggle_PS_20174392719_1491204439457_log.csv --output_path ./task/kaggle_set/x_kaggle.csv --output_path ./task_orig/kaggle_set/x_kaggle.csv --fraud_output ./task_orig/kaggle_set/y_kaggle.csv
    adjust_kaggle()