import click
import importlib
import pandas as pd
import os
import datetime
import logging

# Global static path values (for test data and logs)
TEST_X_PATH = "task/test_set/x_test.csv"
TEST_OUTPUT_PATH = "task/test_set/y_test.csv"
LOG_DIR = "logs"
LOG_BASENAME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"{LOG_BASENAME}.txt")
    
    logger = logging.getLogger("predict_logger")
    logger.setLevel(logging.INFO)
    
    # create console handler and set level to info
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # create file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger

@click.command()
@click.option('--model_module', required=True, help='Python module containing the Model class')
@click.option('--train_x', required=True, multiple=True, help='Path(s) to training X CSV file(s)')
@click.option('--train_y', required=True, multiple=True, help='Path(s) to training Y CSV file(s)')
@click.option('--test_x', default=TEST_X_PATH, help='Path to test X CSV file')
@click.option('--output_path', default=TEST_OUTPUT_PATH, help='Path to write the predictions CSV file')
def main(model_module, train_x, train_y, test_x, output_path):
    """
    This script loads a Model from the given module, concatenates multiple training sources,
    fits the model, and then predicts fraud labels for the test set.
    
    The training set is built by merging each provided pair from --train_x and --train_y 
    on "AccountID" and then appending them together.
    
    The test predictions are written out to the specified output_path with columns:
      AccountID,Fraudster
    """
    logger = setup_logger()
    logger.info("Starting prediction pipeline...")

    # Import the model dynamically.
    try:
        mod = importlib.import_module(model_module)
        Model = getattr(mod, "Model")
        logger.info(f"Imported Model from module {model_module}")
    except Exception as e:
        logger.error(f"Error importing Model from module {model_module}: {e}")
        return

    # Check that the number of provided training X files equals training Y files.
    if len(train_x) != len(train_y):
        logger.error("The number of training X files must equal the number of training Y files.")
        return

    # Load and merge the training data from provided sources.
    merged_dfs = []
    for tx, ty in zip(train_x, train_y):
        logger.info(f"Loading training data from {tx} and {ty}")
        X_df = pd.read_csv(tx)
        Y_df = pd.read_csv(ty)
        merged_df = pd.merge(X_df, Y_df, on="AccountID")
        merged_dfs.append(merged_df)
    all_train = pd.concat(merged_dfs, ignore_index=True)
    logger.info(f"Combined training data: {all_train.shape[0]} samples, {all_train.shape[1]} columns")

    # Split into features and target.
    # Drop the Fraudster column (and AccountID) for features.
    X_train = all_train.drop(columns=["AccountID", "Fraudster"])
    y_train = all_train["Fraudster"]
    
    logger.info(f"Training data prepared: {X_train.shape[0]} samples with {X_train.shape[1]} features.")
    
    # Initialize and train model.
    model = Model()
    logger.info("Training model...")
    model.fit(X_train, y_train)
    logger.info("Model training completed.")

    # --- Predict on test set ---
    logger.info(f"Loading test data from {test_x}")
    x_test = pd.read_csv(test_x)
    test_ids = x_test["AccountID"]
    X_test = x_test.drop(columns=["AccountID"])
    logger.info("Predicting on test set...")
    y_test_pred = model.predict(X_test)
    
    # Prepare output DataFrame.
    out_df = pd.DataFrame({
        "AccountID": test_ids,
        "Fraudster": y_test_pred
    })
    
    # Write predictions to output path.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    logger.info(f"Test predictions written to {output_path}")
    
    # Also, write predictions to logs/y_test.csv.
    logs_test_path = os.path.join(LOG_DIR, f"{LOG_BASENAME}_test.csv")
    out_df.to_csv(logs_test_path, index=False)
    logger.info(f"Test predictions also written to {logs_test_path}")
    logger.info("Prediction pipeline complete.")

if __name__ == "__main__":
    main()