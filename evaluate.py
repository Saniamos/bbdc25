import click
import importlib
import pandas as pd
import os
import datetime
import logging
import gc
from sklearn.metrics import classification_report
from models.aggregates.prediction_threshold import predict_and_aggregate as predict_and_aggregate_threshold
from models.aggregates.proba_threshold import predict_and_aggregate as predict_and_aggregate_proba
from models.aggregates.proba_train import ProbaTrain

# Global static path values
FTSET = '.ver03'
TRAIN_X_PATH = f"task/train_set/x_train{FTSET}.parquet"
TRAIN_Y_PATH = f"task/train_set/y_train{FTSET}.parquet"
VAL_X_PATH   = f"task/val_set/x_val{FTSET}.parquet"
VAL_Y_PATH   = f"task/val_set/y_val{FTSET}.parquet"
# KAGGLE_X_PATH = f"task/kaggle_set/x_kaggle{FTSET}.parquet"
# KAGGLE_Y_PATH = f"task/kaggle_set/y_kaggle{FTSET}.parquet"
TEST_X_PATH  = f"task/test_set/x_test{FTSET}.parquet"
TEST_OUTPUT_PATH = f"task/test_set/y_test{FTSET}.parquet"
SKELETON = "task/professional_skeleton.csv"
LOG_DIR = "logs"
LOG_BASENAME = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

# Combined paths for full retrain
# TRAIN_X_PATHS = [TRAIN_X_PATH, VAL_X_PATH, KAGGLE_X_PATH]
# TRAIN_Y_PATHS = [TRAIN_Y_PATH, VAL_Y_PATH, KAGGLE_Y_PATH]
TRAIN_X_PATHS = [TRAIN_X_PATH, VAL_X_PATH]
TRAIN_Y_PATHS = [TRAIN_Y_PATH, VAL_Y_PATH]

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

def load_training_data(logger):
    """Load and prepare training data"""
    logger.info(f"Loading training data from {TRAIN_X_PATH} and {TRAIN_Y_PATH}")
    x_train = pd.read_parquet(TRAIN_X_PATH)
    y_train_df = pd.read_parquet(TRAIN_Y_PATH)
    
    # Merge training sets on AccountID
    train = pd.merge(x_train, y_train_df, on="AccountID")
    X_train = train.drop(columns=["Fraudster"])
    y_train = train["Fraudster"]
    logger.info(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    return X_train, y_train, y_train_df, x_train

def load_validation_data(logger):
    """Load and prepare validation data"""
    logger.info(f"Loading validation data from {VAL_X_PATH} and {VAL_Y_PATH}")
    x_val_df = pd.read_parquet(VAL_X_PATH)
    y_val_df = pd.read_parquet(VAL_Y_PATH)
    
    val = pd.merge(x_val_df, y_val_df, on="AccountID")
    X_val = val.drop(columns=["Fraudster"])
    y_val = val["Fraudster"]
    logger.info(f"Validation data: {X_val.shape[0]} samples, {X_val.shape[1]} features")
    
    return X_val, y_val, y_val_df, x_val_df

def load_test_data(logger):
    """Load test data"""
    logger.info(f"Loading test data from {TEST_X_PATH}")
    X_test = pd.read_parquet(TEST_X_PATH)
    y_test_df = pd.read_csv(SKELETON)
    test_ids = X_test["AccountID"]
    
    return X_test, y_test_df, test_ids

def load_combined_data(logger):
    """Load and combine all training data with suffixed IDs"""
    logger.info("Loading training data from multiple sources...")
    combined = []
    source_names = ["train", "val", "kaggle"]

    for i, (x_path, y_path) in enumerate(zip(TRAIN_X_PATHS, TRAIN_Y_PATHS)):
        source_suffix = f"_{source_names[i]}"
        logger.info(f"Loading data from {x_path} and {y_path}")
        try:
            x_data = pd.read_parquet(x_path)
            y_data = pd.read_parquet(y_path)
            
            # Add suffix to AccountID to ensure uniqueness
            x_data["AccountID"] = x_data["AccountID"].astype(str) + source_suffix
            non_empty_external = ~x_data["External"].isna()
            x_data.loc[non_empty_external, "External"] = x_data.loc[non_empty_external, "External"].astype(str) + source_suffix
            y_data["AccountID"] = y_data["AccountID"].astype(str) + source_suffix
            
            # Merge sets on AccountID
            dataset = pd.merge(x_data, y_data, on="AccountID")
            combined.append(dataset)
            logger.info(f"Successfully loaded {len(dataset)} samples from {x_path} with AccountID suffix '{source_suffix}'")
            
            # Free memory
            del x_data, y_data, dataset
            gc.collect()
        except Exception as e:
            logger.warning(f"Could not load data from {x_path} or {y_path}: {e}")

    # Combine all datasets
    logger.info("Combining all available training data...")
    combined_data = pd.concat(combined, ignore_index=True)
    X_combined = combined_data.drop(columns=["Fraudster"])
    y_combined = combined_data["Fraudster"]
    y_combined_df = combined_data[["AccountID", "Fraudster"]]
    logger.info(f"Combined data: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")
    
    # Free memory
    del combined
    gc.collect()
    
    return X_combined, y_combined, y_combined_df, combined_data

def train_model(Model, X, y, logger, save_path=None):
    """Train and optionally save a model"""
    model = Model()
    logger.info(f"Model hyperparameters: {model.log_params()}")
    cur_time = datetime.datetime.now()
    logger.info("Training model...")
    model.fit(X, y)
    logger.info(f"Model training completed. Took: {datetime.datetime.now() - cur_time}")
    
    if save_path:
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")
    
    return model

def evaluate_model(model, X_val, y_val, logger):
    """Evaluate model on validation data"""
    logger.info("Predicting on validation set...")
    y_val_pred = model.predict(X_val)
    report = classification_report(y_val, y_val_pred)
    logger.info("Classification Report on Validation Set:\n" + report)
    return y_val_pred

def evaluate_account_aggregations(model, x_val_df, y_val_df, logger):
    """Evaluate different account aggregation methods"""
    methods = [
        ("predict_and_aggregate_threshold", predict_and_aggregate_threshold, dict(method='threshold')),
        ("predict_and_aggregate_proba", predict_and_aggregate_proba, {}),
        ("predict_and_aggregate_threshold", predict_and_aggregate_threshold, dict(method='threshold_sum')),
        ("predict_and_aggregate_threshold", predict_and_aggregate_threshold, dict(method='mean'))
    ]
    
    for name, method, params in methods:
        logger.info(f"Classification Report **Per Account** on Validation Set:\n{name}({params})")
        accountids_val = method(model, x_val_df, logger=logger, **params)
        accountids_val = accountids_val.set_index('AccountID').reindex(y_val_df['AccountID']).reset_index()
        assert all(accountids_val['AccountID'] == y_val_df['AccountID'])
        report = classification_report(y_val_df['Fraudster'], accountids_val['Fraudster'])
        logger.info(report)
        logger.info('--------------------------------------------------')
        
        # Free memory
        del accountids_val
        gc.collect()

def evaluate_proba_train(model, x_train, y_train_df, x_val_df, y_val_df, logger):
    """Evaluate ProbaTrain on validation data"""
    logger.info("Classification Report **Per Account** on Validation Set with ProbaTrain")
    proba_train = ProbaTrain(model=model, logger=logger)
    proba_train.fit(x_train, y_train_df)
    accountids_val_proba_train = proba_train.predict(x_val_df, skeleton=y_val_df)
    report = classification_report(y_val_df['Fraudster'], accountids_val_proba_train['Fraudster'])
    logger.info(report)
    logger.info('--------------------------------------------------')
    
    return proba_train

def predict_and_save(model, X_test, test_ids, output_path, logger):
    """Make predictions and save results"""
    logger.info("Predicting on test set...")
    y_test_pred = model.predict(X_test)
    out_df = pd.DataFrame({
        "AccountID": test_ids,
        "Fraudster": y_test_pred
    })
    out_df.to_csv(output_path, index=False)
    logger.info(f"Test predictions written to {output_path}")
    
    # Free memory
    del out_df, y_test_pred
    gc.collect()

def predict_aggregate_and_save(model, X_test, y_test_df, output_path, logger, method='threshold'):
    """Predict with aggregation and save results"""
    accountids = predict_and_aggregate_threshold(model, X_test, method=method, logger=logger)
    accountids = accountids.set_index('AccountID').reindex(y_test_df['AccountID']).reset_index()
    assert all(accountids['AccountID'] == y_test_df['AccountID'])
    accountids.to_csv(output_path, index=False)
    logger.info(f"Test account id predictions written to {output_path}")
    
    # Free memory
    del accountids
    gc.collect()

def proba_train_predict_and_save(proba_train, X_test, y_test_df, output_path, logger):
    """Predict using ProbaTrain and save results"""
    accountids_proba_train = proba_train.predict(X_test, skeleton=y_test_df)
    accountids_proba_train.to_csv(output_path, index=False)
    logger.info(f"Test account id predictions with ProbaTrain written to {output_path}")
    
    # Free memory
    del accountids_proba_train
    gc.collect()

# Add this new function 
def refit_model(model, X, y, logger, save_path=None):
    """Refit an existing model with new data and optionally save it"""
    cur_time = datetime.datetime.now()
    logger.info("Refitting model...")
    model.refit(X, y)
    logger.info(f"Model refitting completed. Took: {datetime.datetime.now() - cur_time}")
    
    if save_path:
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")
    
    return model

@click.command()
@click.option('--model_module', default="models.guess", required=True, help='Python module containing the Model class')
@click.option('--retrain', is_flag=True, default=False, help='If set, retrain on all data after evaluation and predict test set')
def main(model_module, retrain):
    """Main function with evaluation and optional full retrain"""
    global LOG_BASENAME
    LOG_BASENAME += f"_{model_module}"
    if retrain:
        LOG_BASENAME += "_retrain"

    logger = setup_logger()
    # Log configuration
    logger.info(f"FTSET: {FTSET}")
    logger.info(f"Full Retrain: {retrain}")
    logger.info(f"Model Module: {model_module}")
    
    # Dynamically import the Model class
    try:
        mod = importlib.import_module(model_module)
        Model = getattr(mod, "Model")
        logger.info(f"Imported Model from module {model_module}")
    except Exception as e:
        logger.error(f"Error importing Model from module {model_module}: {e}")
        return

    # --- Standard evaluation process ---
    # Load training data
    X_train, y_train, y_train_df, x_train = load_training_data(logger)
    
    # Train model on training data
    model_save_path = os.path.join(LOG_DIR, f"{LOG_BASENAME}_model.pkl")
    model = train_model(Model, X_train, y_train, logger, save_path=model_save_path)
    
    # Load validation data
    X_val, y_val, y_val_df, x_val_df = load_validation_data(logger)
    
    # Evaluate model
    evaluate_model(model, X_val, y_val, logger)
    
    # Evaluate account aggregations
    evaluate_account_aggregations(model, x_val_df, y_val_df, logger)
    
    # Evaluate with ProbaTrain
    proba_train = evaluate_proba_train(model, x_train, y_train_df, x_val_df, y_val_df, logger)
    
    # Free training and validation data memory
    del X_train, y_train, y_train_df, x_train, X_val, y_val, y_val_df, x_val_df
    gc.collect()
    
    # Load test data
    X_test, y_test_df, test_ids = load_test_data(logger)
    
    # Make transaction-level predictions
    logs_test_path = os.path.join(LOG_DIR, f"{LOG_BASENAME}_test_transaction.csv")
    predict_and_save(model, X_test, test_ids, logs_test_path, logger)
    
    # Make account-level predictions with threshold
    logs_test_path = os.path.join(LOG_DIR, f"{LOG_BASENAME}_test.csv")
    predict_aggregate_and_save(model, X_test, y_test_df, logs_test_path, logger)
    
    # Make account-level predictions with ProbaTrain
    logs_test_proba_path = os.path.join(LOG_DIR, f"{LOG_BASENAME}_test_proba_train.csv")
    proba_train_predict_and_save(proba_train, X_test, y_test_df, logs_test_proba_path, logger)
    
    logger.info("Standard evaluation complete.")
    logger.info("===============================================")
    
    # Free memory from standard evaluation
    del proba_train, X_test, test_ids
    gc.collect()
    
    # Replace the full retrain section with this updated code:

    # --- Full retrain process (if flag is set) ---
    if retrain:
        logger.info("Starting full retrain process using model.refit()...")
        
        # Load and combine all training data
        X_combined, y_combined, y_combined_df, combined_data = load_combined_data(logger)
        
        # Refit existing model on combined data instead of training from scratch
        full_model_save_path = os.path.join(LOG_DIR, f"{LOG_BASENAME}_full_model.pkl")
        full_model = refit_model(model, X_combined, y_combined, logger, save_path=full_model_save_path)
        
        # Load test data
        X_test_full, y_test_df_full, test_ids_full = load_test_data(logger)
        
        # Make transaction-level predictions
        logs_test_full_path = os.path.join(LOG_DIR, f"{LOG_BASENAME}_full_test_transaction.csv")
        predict_and_save(full_model, X_test_full, test_ids_full, logs_test_full_path, logger)
        
        # Make account-level predictions with threshold
        logs_test_full_path = os.path.join(LOG_DIR, f"{LOG_BASENAME}_full_test.csv")
        predict_aggregate_and_save(full_model, X_test_full, y_test_df_full, logs_test_full_path, logger)
        
        # Train ProbaTrain on combined data
        logger.info("Training ProbaTrain on all data for final predictions...")
        proba_train_full = ProbaTrain(model=full_model, logger=logger)
        proba_train_full.fit(X_combined, y_combined_df)
        
        # Make account-level predictions with ProbaTrain
        logs_test_proba_full_path = os.path.join(LOG_DIR, f"{LOG_BASENAME}_full_test_proba_train.csv")
        proba_train_predict_and_save(proba_train_full, X_test_full, y_test_df_full, logs_test_proba_full_path, logger)
        
        logger.info("Full retrain and prediction process complete.")
        
        # Free memory from full retrain
        del X_combined, y_combined, y_combined_df, combined_data, full_model, proba_train_full
        gc.collect()

if __name__ == "__main__":
    # example usage:
    # python3 evaluate.py --model_module models.rf
    # python3 evaluate.py --model_module models.rf --retrain
    main()