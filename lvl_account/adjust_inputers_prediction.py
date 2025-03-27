import click
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report

# New helper function to add inputer scores from transactions
def add_inputer_scores(pred_df, trans_df):
    # Vectorized approach to compute inputer fraud scores
    # Merge trans_df with pred_df to get inputer fraud scores from each transaction
    merged = pd.merge(trans_df[['External', 'AccountID', 'Amount']],
                      pred_df[['AccountID', 'Fraudster_pred']],
                      on="AccountID", how="left")
    agg = merged.groupby('External')[['Fraudster_pred', 'Amount']].agg(['mean', 'max', 'min', 'std']).reset_index()
    agg.columns = [col if isinstance(col, str) else f"{col[0]}_{col[1]}" for col in agg.columns]
    agg.rename({'External_': 'External'}, axis=1, inplace=True)
    pred_df = pred_df.merge(agg, left_on="AccountID", right_on='External', how="left", suffixes=['', '_agg'])
    cols = pred_df.drop(columns=['AccountID', 'External', 'Fraudster_pred']).columns
    pred_df[cols] = pred_df[cols].fillna(0)
    return pred_df, cols

@click.command()
@click.option("--data_version", default="ver00", type=str, help="Data version to use, since we're only calculating inputers this does not make a difference atm.")
@click.argument('pred_csv_base', default='/home/yale/Repositories/bbdc25/lvl_account/logs/2025.03.26_12.24.12_attn_cnn', type=str)
def main(data_version, pred_csv_base):
    # Infer prediction CSVs for each stage and corresponding true and transaction files
    pred_train_csv = f"{pred_csv_base}_train.csv"
    pred_val_csv   = f"{pred_csv_base}_val.csv"
    pred_test_csv  = f"{pred_csv_base}_test.csv"

    train_true_csv    = f"/home/yale/Repositories/bbdc25/task/train_set/y_train.{data_version}.parquet"
    val_true_csv      = f"/home/yale/Repositories/bbdc25/task/val_set/y_val.{data_version}.parquet"
    train_trans_file  = f"/home/yale/Repositories/bbdc25/task/train_set/x_train.{data_version}.parquet"
    val_trans_file    = f"/home/yale/Repositories/bbdc25/task/val_set/x_val.{data_version}.parquet"
    test_trans_file   = f"/home/yale/Repositories/bbdc25/task/test_set/x_test.{data_version}.parquet"
    
    click.echo(f"Using data_version: {data_version}")
    click.echo(f"Loading train predictions from: {pred_train_csv}")
    train_df = pd.read_csv(pred_train_csv).rename({'Fraudster': 'Fraudster_pred'}, axis=1)
    click.echo(f"Loading val predictions from: {pred_val_csv}")
    val_df = pd.read_csv(pred_val_csv).rename({'Fraudster': 'Fraudster_pred'}, axis=1)
    click.echo(f"Loading test predictions from: {pred_test_csv}")
    test_df = pd.read_csv(pred_test_csv).rename({'Fraudster': 'Fraudster_pred'}, axis=1)
    
    # Load true labels for train and val
    true_train = pd.read_parquet(train_true_csv).rename({"Fraudster": "Fraudster_true"}, axis=1)
    true_val   = pd.read_parquet(val_true_csv).rename({"Fraudster": "Fraudster_true"}, axis=1)
    click.echo(f"Loaded train true labels with {len(true_train)} records.")
    click.echo(f"Loaded val true labels with {len(true_val)} records.")
    
    # Merge true labels into train and val DataFrames
    train_df = train_df.merge(true_train[['AccountID', 'Fraudster_true']], on="AccountID", how="left")
    val_df = val_df.merge(true_val[['AccountID', 'Fraudster_true']], on="AccountID", how="left")
    
    # Load transactions files for each stage
    train_trans = pd.read_parquet(train_trans_file)
    val_trans   = pd.read_parquet(val_trans_file)
    test_trans  = pd.read_parquet(test_trans_file)
    click.echo(f"Loaded train transactions with {len(train_trans)} records.")
    click.echo(f"Loaded val transactions with {len(val_trans)} records.")
    click.echo(f"Loaded test transactions with {len(test_trans)} records.")
    
    # Add inputer fraud scores to each stage
    val_df, cols   = add_inputer_scores(val_df, val_trans)
    train_df, cols = add_inputer_scores(train_df, train_trans)
    test_df, cols  = add_inputer_scores(test_df, test_trans)
    click.echo("Added highest and lowest inputer fraud scores for all stages.")
    
    val_df['Miss'] = val_df['Fraudster_pred'] != val_df['Fraudster_true']
    print(val_df.sort_values(['Miss'], ascending=False)[:20])
    # Prepare features and target (for train and val)
    # features = ["Fraudster_pred", "Fraudster_pred_agg"]
    features = ["Fraudster_pred"] + list(cols)
    target = "Fraudster_true"
    
    # Ensure that train and val data have complete records
    train_df = train_df.dropna(subset=features + [target])
    val_df = val_df.dropna(subset=features + [target])
    if train_df.empty or val_df.empty:
        click.echo("Insufficient records with complete inputer fraud scores and true labels. Exiting.")
        return

    # Use train and val directly without splitting
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    
    # Initialize and train the XGBoost classifier on train data
    clf = xgb.XGBClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate on the validation set
    y_pred_val = clf.predict(X_val)
    click.echo("Validation Set Classification Report:")
    print(classification_report(y_val, y_pred_val))
    
    # # Predict on the test set and output summary
    # X_test = test_df[features].dropna()  # In case of any missing values
    # if X_test.empty:
    #     click.echo("No valid records in test set for prediction. Exiting.")
    #     return
    # test_df["predicted"] = clf.predict(X_test)
    # click.echo("Test predictions completed. Sample output:")
    # click.echo(test_df[["AccountID", "predicted"]].head().to_string(index=False))
    
if __name__ == '__main__':
    main()