from functools import partial
import click
import importlib
import pandas as pd
import os
import datetime
import logging
from sklearn.metrics import classification_report
from models.aggregates.prediction_threshold import predict_and_aggregate as predict_and_aggregate_threshold
from models.aggregates.proba_threshold import predict_and_aggregate as predict_and_aggregate_proba
from models.aggregates.proba_train import ProbaTrain

# Global static path values
FTSET = ''
FTSET = '.ver01'
FTSET = '.ver02'
TRAIN_X_PATH = f"task/train_set/x_train{FTSET}.parquet"
TRAIN_Y_PATH = f"task/train_set/y_train{FTSET}.parquet"
VAL_X_PATH   = f"task/val_set/x_val{FTSET}.parquet"
VAL_Y_PATH   = f"task/val_set/y_val{FTSET}.parquet"
TEST_X_PATH  = f"task/test_set/x_test{FTSET}.parquet"
TEST_OUTPUT_PATH = f"task/test_set/y_test{FTSET}.parquet"
SKELETON = "task/professional_skeleton.csv"
LOG_DIR = "logs"
LOG_BASENAME = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")

# Removed: local transaction_to_accountid function

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
@click.option('--model_module', default="models.brf", required=True, help='Python module containing the Model class')
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
    x_train = pd.read_parquet(TRAIN_X_PATH)
    y_train_df = pd.read_parquet(TRAIN_Y_PATH)  # Keep original DataFrame for ProbaTrain
    # Merge training sets on AccountID
    train = pd.merge(x_train, y_train_df, on="AccountID")
    # Features: drop AccountID and Fraudster.
    X_train = train.drop(columns=["Fraudster"])
    y_train = train["Fraudster"]
    logger.info(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # --- Initialize and train model ---
    model = Model()
    logger.info(f"Model hyperparameters: {model.log_params()}")
    cur_time = datetime.datetime.now()
    logger.info("Training model...")
    model.fit(X_train, y_train)
    logger.info(f"Model training completed. Took: {datetime.datetime.now() - cur_time} min")
    model.save(os.path.join(LOG_DIR, f"{LOG_BASENAME}_model.pkl"))

    # --- Load validation data and evaluate ---
    logger.info(f"Loading validation data from {VAL_X_PATH} and {VAL_Y_PATH}")
    x_val_df = pd.read_parquet(VAL_X_PATH)
    y_val_df = pd.read_parquet(VAL_Y_PATH)
    val = pd.merge(x_val_df, y_val_df, on="AccountID")
    X_val = val.drop(columns=["Fraudster"])
    y_val = val["Fraudster"]
    logger.info(f"Validation data: {X_val.shape[0]} samples, {X_val.shape[1]} features")
    
    logger.info("Predicting on validation set...")
    y_val_pred = model.predict(X_val)
    report = classification_report(y_val, y_val_pred)
    logger.info("Classification Report on Validation Set:\n" + report)

    # Use new aggregated prediction for validation
    for name, method, params in [("predict_and_aggregate_threshold", predict_and_aggregate_threshold, dict(method='threshold')),
                    ("predict_and_aggregate_proba", predict_and_aggregate_proba, {}),
                    ("predict_and_aggregate_threshold", predict_and_aggregate_threshold, dict(method='threshold_sum')),
                    ("predict_and_aggregate_threshold", predict_and_aggregate_threshold, dict(method='mean'))]:
        logger.info(f"Classification Report **Per Account** on Validation Set:\n{name}({params})")
        accountids_val = method(model, x_val_df, logger=logger, **params)
        accountids_val = accountids_val.set_index('AccountID').reindex(y_val_df['AccountID']).reset_index()
        assert all(accountids_val['AccountID'] == y_val_df['AccountID'])
        report = classification_report(y_val_df['Fraudster'], accountids_val['Fraudster'])
        logger.info(report)
        logger.info('--------------------------------------------------')
    
    # Use ProbaTrain for validation
    logger.info("Classification Report **Per Account** on Validation Set with ProbaTrain")
    proba_train = ProbaTrain(model=model, logger=logger)
    proba_train.fit(x_train, y_train_df)
    accountids_val_proba_train = proba_train.predict(x_val_df, skeleton=y_val_df)
    report = classification_report(y_val_df['Fraudster'], accountids_val_proba_train['Fraudster'])
    logger.info(report)
    logger.info('--------------------------------------------------')

    # --- Predict on test set and output predictions ---
    logger.info(f"Loading test data from {TEST_X_PATH}")
    X_test = pd.read_parquet(TEST_X_PATH)
    y_test_df = pd.read_csv(SKELETON)
    # Preserve AccountID for output.
    test_ids = X_test["AccountID"]
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

    # Use new aggregated prediction for test
    accountids = predict_and_aggregate_threshold(model, X_test, method='threshold', logger=logger)
    accountids = accountids.set_index('AccountID').reindex(y_test_df['AccountID']).reset_index()
    assert all(accountids['AccountID'] == y_test_df['AccountID'])
    logs_test_path = os.path.join(LOG_DIR, f"{LOG_BASENAME}_test.csv")
    accountids.to_csv(logs_test_path, index=False)
    logger.info(f"Test account id predictions also written to {logs_test_path}")
    
    # Use ProbaTrain for test predictions
    accountids_proba_train = proba_train.predict(X_test, skeleton=y_test_df)
    logs_test_proba_path = os.path.join(LOG_DIR, f"{LOG_BASENAME}_test_proba_train.csv")
    accountids_proba_train.to_csv(logs_test_proba_path, index=False)
    logger.info(f"Test account id predictions with ProbaTrain written to {logs_test_proba_path}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    # example useage:
    # python evaluate.py --model_module models.rf
    main()