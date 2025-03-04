import click
import importlib
import pandas as pd
import os
import datetime
import logging
from sklearn.metrics import classification_report

# Global static path values
FTSET = ''
FTSET = '.ver01'
TRAIN_X_PATH = "task/train_set/x_train{FTSET}.csv"
TRAIN_Y_PATH = "task/train_set/y_train{FTSET}.csv"
VAL_X_PATH   = "task/val_set/x_val{FTSET}.csv"
VAL_Y_PATH   = "task/val_set/y_val{FTSET}.csv"
TEST_X_PATH  = "task/test_set/x_test{FTSET}.csv"
TEST_OUTPUT_PATH = "task/test_set/y_test{FTSET}.csv"
SKELETON = "task/professional_skeleton.csv"
LOG_DIR = "logs"
LOG_BASENAME = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")

def transaction_to_accountid(df, col='Fraudster', method='threshold'):
    if method == 'mean':
        # simple majority vote
        return df.groupby('AccountID')[col].mean().round().astype(int).reset_index()
    
    if method == 'threshold':
        # threshold based
        grp = df.groupby('AccountID')[col].mean()
        threshold = grp.quantile(0.85)  # Select top 15%
        grp = (grp >= threshold).astype(int).reset_index()
        return grp
    
    if method == 'threshold_sum':
        # threshold based
        grp = df.groupby('AccountID')[col].sum()
        threshold = grp.quantile(0.85)  # Select top 15%
        grp = (grp >= threshold).astype(int).reset_index()
        return grp

# Set up logging
def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"{LOG_BASENAME}.txt")
    
    logger = logging.getLogger("evaluate_logger")
    logger.setLevel(logging.INFO)
    
    # create console handler and set level to info
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger

@click.command()
@click.option('--model_module', default="models.rf", required=True, help='Python module containing the Model class')
def main(model_module):
    """
    This script loads a Model from the given module, trains it on the training set,
    evaluates it on the validation set (printing a classification report to stdout),
    and then uses it to predict fraud labels for the test set.
    
    Expected data files:
      - Training: TRAIN_X_PATH, TRAIN_Y_PATH
      - Validation: VAL_X_PATH, VAL_Y_PATH
      - Test: TEST_X_PATH
         
    The test predictions are written out to TEST_OUTPUT_PATH (with columns AccountID,Fraudster).
    All log output is written to stdout and also saved to logs/<timestamp>.txt.
    """
    global LOG_BASENAME
    LOG_BASENAME += f"_{model_module}"

    logger = setup_logger()
    # log configuration
    logger.info(f"FTSET: {FTSET}")
    logger.info(f"TRAIN_X_PATH: {TRAIN_X_PATH}")
    logger.info(f"TRAIN_Y_PATH: {TRAIN_Y_PATH}")
    logger.info(f"VAL_X_PATH: {VAL_X_PATH}")
    logger.info(f"VAL_Y_PATH: {VAL_Y_PATH}")
    logger.info(f"TEST_X_PATH: {TEST_X_PATH}")
    logger.info(f"TEST_OUTPUT_PATH: {TEST_OUTPUT_PATH}")
    # start evaluation
    logger.info("Starting evaluation...")

    # Dynamically import the module containing the Model class.
    try:
        mod = importlib.import_module(model_module)
        Model = getattr(mod, "Model")
        logger.info(f"Imported Model from module {model_module}")
    except Exception as e:
        logger.error(f"Error importing Model from module {model_module}: {e}")
        return

    # --- Load training data ---
    logger.info(f"Loading training data from {TRAIN_X_PATH} and {TRAIN_Y_PATH}")
    x_train = pd.read_csv(TRAIN_X_PATH, compression='gzip')
    y_train = pd.read_csv(TRAIN_Y_PATH, compression='infer')
    # Merge training sets on AccountID
    train = pd.merge(x_train, y_train, on="AccountID")
    # Features: drop AccountID and Fraudster.
    X_train = train.drop(columns=["Fraudster"])
    y_train = train["Fraudster"]
    logger.info(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # --- Initialize and train model ---
    model = Model()
    cur_time = datetime.datetime.now()
    logger.info("Training model...")
    model.fit(X_train, y_train)
    logger.info(f"Model training completed. Took: {datetime.datetime.now() - cur_time} min")
    model.save(os.path.join(LOG_DIR, f"{LOG_BASENAME}_model.pkl"))

    # --- Load validation data and evaluate ---
    logger.info(f"Loading validation data from {VAL_X_PATH} and {VAL_Y_PATH}")
    x_val_df = pd.read_csv(VAL_X_PATH, compression='gzip')
    y_val_df = pd.read_csv(VAL_Y_PATH, compression='infer')
    val = pd.merge(x_val_df, y_val_df, on="AccountID")
    X_val = val.drop(columns=["Fraudster"])
    y_val = val["Fraudster"]
    logger.info(f"Validation data: {X_val.shape[0]} samples, {X_val.shape[1]} features")
    
    logger.info("Predicting on validation set...")
    y_val_pred = model.predict(X_val)
    report = classification_report(y_val, y_val_pred)
    logger.info("Classification Report on Validation Set:\n" + report)

    # Additionally calc per AccountID predictions.
    x_val_df['Pred'] = y_val_pred
    accountids_val = transaction_to_accountid(x_val_df, col='Pred')
    accountids_val = accountids_val.set_index('AccountID').reindex(y_val_df['AccountID']).reset_index()
    assert all(accountids_val['AccountID'] == y_val_df['AccountID'])
    report = classification_report(y_val_df['Fraudster'], accountids_val['Pred'])
    logger.info("Classification Report **Per Account** on Validation Set:\n" + report)

    # --- Predict on test set and output predictions ---
    logger.info(f"Loading test data from {TEST_X_PATH}")
    x_test = pd.read_csv(TEST_X_PATH, compression='gzip')
    y_test_df = pd.read_csv(SKELETON, compression='infer')
    # Preserve AccountID for output.
    test_ids = x_test["AccountID"]
    X_test = x_test
    # X_test = x_test.drop(columns=[])
    logger.info("Predicting on test set...")
    y_test_pred = model.predict(X_test)
    # Prepare output DataFrame with columns AccountID, Fraudster.
    out_df = pd.DataFrame({
        "AccountID": test_ids,
        "Fraudster": y_test_pred
    })
    # Write the output DataFrame to logs/y_test.csv.
    logs_test_path = os.path.join(LOG_DIR, f"{LOG_BASENAME}_test_transaction.csv")
    out_df.to_csv(logs_test_path, index=False)
    logger.info(f"Test predictions also written to {logs_test_path}")

    # Additionally write out per AccountID predictions.
    accountids = transaction_to_accountid(out_df, col='Fraudster')
    accountids = accountids.set_index('AccountID').reindex(y_test_df['AccountID']).reset_index()
    assert all(accountids['AccountID'] == y_test_df['AccountID'])
    logs_test_path = os.path.join(LOG_DIR, f"{LOG_BASENAME}_test.csv")
    accountids.to_csv(logs_test_path, index=False)
    logger.info(f"Test account id predictions also written to {logs_test_path}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    # example useage:
    # python evaluate.py --model_module models.rf
    main()