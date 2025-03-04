import click
import importlib
import pandas as pd
import os
import datetime
import logging
from models.aggregates.prediction_threshold import predict_and_aggregate

# Global static path values
FTSET = ''
FTSET = '.ver01'
TRAIN_X_PATHS = [f"task/train_set/x_train{FTSET}.parquet", f"task/val_set/x_val{FTSET}.parquet", f"task/kaggle_set/x_kaggle{FTSET}.parquet"]
TRAIN_Y_PATHS = [f"task/train_set/y_train{FTSET}.parquet", f"task/val_set/y_val{FTSET}.parquet", f"task/kaggle_set/y_kaggle{FTSET}.parquet"]
TEST_X_PATH  = f"task/test_set/x_test{FTSET}.parquet"
TEST_OUTPUT_PATH = f"task/test_set/y_test{FTSET}.parquet"
SKELETON = "task/professional_skeleton.csv"
LOG_DIR = "logs"
LOG_BASENAME = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")

# Set up logging
def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"{LOG_BASENAME}.txt")
    
    logger = logging.getLogger("predict_logger")
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
    This script loads a Model from the given module, trains it on the combined training and validation sets,
    and then uses it to predict fraud labels for the test set.
    
    Expected data files:
      - Training: TRAIN_X_PATH, TRAIN_Y_PATH
      - Validation: VAL_X_PATH, VAL_Y_PATH
      - Test: TEST_X_PATH
         
    The test predictions are written out to the logs directory with the format:
    logs/<timestamp>_<model_module>_test.csv (with columns AccountID,Fraudster).
    All log output is written to stdout and also saved to logs/<timestamp>_<model_module>.txt.
    """
    global LOG_BASENAME
    LOG_BASENAME += f"_{model_module}"

    logger = setup_logger()
    logger.info("Starting prediction process...")

    # Dynamically import the module containing the Model class.
    try:
        mod = importlib.import_module(model_module)
        Model = getattr(mod, "Model")
        logger.info(f"Imported Model from module {model_module}")
    except Exception as e:
        logger.error(f"Error importing Model from module {model_module}: {e}")
        return
    
    # --- Load and combine all training data ---
    logger.info("Loading training data from multiple sources...")
    combined = []

    for x_path, y_path in zip(TRAIN_X_PATHS, TRAIN_Y_PATHS):
        logger.info(f"Loading data from {x_path} and {y_path}")
        try:
            x_data = pd.read_parquet(x_path)
            y_data = pd.read_parquet(y_path)
            # Merge sets on AccountID
            dataset = pd.merge(x_data, y_data, on="AccountID")
            combined.append(dataset)
            logger.info(f"Successfully loaded {len(dataset)} samples from {x_path}")
        except Exception as e:
            logger.warning(f"Could not load data from {x_path} or {y_path}: {e}")

    # --- Combine all datasets ---
    logger.info("Combining all available training data...")
    combined_data = pd.concat(combined, ignore_index=True)
    X_combined = combined_data.drop(columns=["Fraudster"])
    y_combined = combined_data["Fraudster"]
    logger.info(f"Combined data: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")
    
    # --- Initialize and train model ---
    model = Model()
    cur_time = datetime.datetime.now()
    logger.info("Training model on combined data...")
    model.fit(X_combined, y_combined)
    logger.info(f"Model training completed. Took: {datetime.datetime.now() - cur_time}")
    model.save(os.path.join(LOG_DIR, f"{LOG_BASENAME}_model.pkl"))

    # --- Predict on test set and output predictions ---
    logger.info(f"Loading test data from {TEST_X_PATH}")
    x_test = pd.read_parquet(TEST_X_PATH)
    y_test_df = pd.read_csv(SKELETON)
    test_ids = x_test["AccountID"]
    logger.info("Predicting on test set...")
    y_test_pred = model.predict(x_test)
    out_df = pd.DataFrame({
        "AccountID": test_ids,
        "Fraudster": y_test_pred
    })
    logs_test_path = os.path.join(LOG_DIR, f"{LOG_BASENAME}_test_transaction.csv")
    out_df.to_csv(logs_test_path, index=False)
    logger.info(f"Test transaction predictions written to {logs_test_path}")

    # Use the new aggregation interface
    accountids = predict_and_aggregate(model, x_test, method='threshold', logger=logger)
    accountids = accountids.set_index('AccountID').reindex(y_test_df['AccountID']).reset_index()
    assert all(accountids['AccountID'] == y_test_df['AccountID'])
    logs_test_path = os.path.join(LOG_DIR, f"{LOG_BASENAME}_test.csv")
    accountids.to_csv(logs_test_path, index=False)
    logger.info(f"Test account predictions written to {logs_test_path}")

    logger.info("Prediction process complete.")


if __name__ == "__main__":
    # example usage:
    # python predict.py --model_module models.rf
    main()