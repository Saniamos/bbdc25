from functools import partial
import click
import pandas as pd
import os
import datetime
import logging
from sklearn.metrics import classification_report
import importlib
import gc
import numpy as np
import psutil
import torch
from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.utils.results_visualizer import PlotSettingParams
from models.aggregates.prediction_threshold import predict_and_aggregate as predict_and_aggregate_threshold
from models.aggregates.proba_threshold import predict_and_aggregate as predict_and_aggregate_proba
from models.aggregates.proba_train import ProbaTrain

# Global static path values (same as evaluate.py)
FTSET = ''
FTSET = '.ver01'
FTSET = '.ver02'
TRAIN_X_PATH = f"task/train_set/x_train{FTSET}.parquet"
TRAIN_Y_PATH = f"task/train_set/y_train{FTSET}.parquet"
VAL_X_PATH   = f"task/val_set/x_val{FTSET}.parquet"
VAL_Y_PATH   = f"task/val_set/y_val{FTSET}.parquet"
KAGGLE_X_PATH = f"task/kaggle_set/x_kaggle{FTSET}.parquet"
KAGGLE_Y_PATH = f"task/kaggle_set/y_kaggle{FTSET}.parquet"
TEST_X_PATH  = f"task/test_set/x_test{FTSET}.parquet"
TEST_OUTPUT_PATH = f"task/test_set/y_test{FTSET}.parquet"
SKELETON = "task/professional_skeleton.csv"
LOG_DIR = "logs"
LOG_BASENAME = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")

# Combined paths for full retrain
TRAIN_X_PATHS = [TRAIN_X_PATH, VAL_X_PATH, KAGGLE_X_PATH]
TRAIN_Y_PATHS = [TRAIN_Y_PATH, VAL_Y_PATH, KAGGLE_Y_PATH]

def free_memory():
    """Force garbage collection and free up memory."""
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

def log_memory_usage(logger, prefix=""):
    """Log the current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"{prefix} Memory Usage: {memory_info.rss / (1024 ** 2):.2f} MB")

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

# AutoPyTorch Model wrapper to match the interface expected in evaluate.py
class AutoPyTorchModel:
    def __init__(self, optimize_metric='f1', walltime_limit=300, eval_time_limit=50):
        self.optimize_metric = optimize_metric
        self.walltime_limit = walltime_limit
        self.eval_time_limit = eval_time_limit
        self.api = TabularClassificationTask()
        
    def fit(self, X, y):
        # Ensure AccountID is not used in training
        X_train = X.drop('AccountID', axis=1) if 'AccountID' in X.columns else X
        self.api.search(
            X_train=X_train,
            y_train=y,
            optimize_metric=self.optimize_metric,
            total_walltime_limit=self.walltime_limit,
            func_eval_time_limit_secs=self.eval_time_limit
        )
        return self
    
    def predict(self, X):
        X_pred = X.drop('AccountID', axis=1) if 'AccountID' in X.columns else X
        return self.api.predict(X_pred)
    
    def predict_proba(self, X):
        X_pred = X.drop('AccountID', axis=1) if 'AccountID' in X.columns else X
        return self.api.predict_proba(X_pred)
    
    def score(self, X, y):
        X_score = X.drop('AccountID', axis=1) if 'AccountID' in X.columns else X
        y_pred = self.predict(X_score)
        return self.api.score(y_pred, y)
    
    def save(self, path):
        # Extract directory path
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        # Save the model
        self.api.save_models(path)
        return True
        
    def load(self, path):
        self.api.load_models(path)
        return self
    
    def log_params(self):
        return {
            "optimize_metric": self.optimize_metric,
            "walltime_limit": self.walltime_limit,
            "eval_time_limit": self.eval_time_limit
        }
    
    def plot_perf_over_time(self, output_path, show=False):
        params = PlotSettingParams(
            xscale='log',
            xlabel='Runtime',
            ylabel=self.optimize_metric,
            title=f'AutoPyTorch Performance ({self.optimize_metric})',
            figname=output_path,
            savefig_kwargs={'bbox_inches': 'tight'},
            show=show
        )
        
        self.api.plot_perf_over_time(
            metric_name=self.optimize_metric,
            plot_setting_params=params
        )

def load_train_data(logger):
    """Load training data."""
    logger.info(f"Loading training data from {TRAIN_X_PATH} and {TRAIN_Y_PATH}")
    x_train = pd.read_parquet(TRAIN_X_PATH)
    y_train_df = pd.read_parquet(TRAIN_Y_PATH)
    # Merge training sets on AccountID
    train = pd.merge(x_train, y_train_df, on="AccountID")
    # Features: drop AccountID and Fraudster.
    X_train = train.drop(columns=["Fraudster"])
    y_train = train["Fraudster"]
    logger.info(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    # Free memory
    del x_train, y_train_df, train
    free_memory()
    
    return X_train, y_train

def load_validation_data(logger):
    """Load validation data."""
    logger.info(f"Loading validation data from {VAL_X_PATH} and {VAL_Y_PATH}")
    x_val_df = pd.read_parquet(VAL_X_PATH)
    y_val_df = pd.read_parquet(VAL_Y_PATH)
    val = pd.merge(x_val_df, y_val_df, on="AccountID")
    X_val = val.drop(columns=["Fraudster"])
    y_val = val["Fraudster"]
    logger.info(f"Validation data: {X_val.shape[0]} samples, {X_val.shape[1]} features")
    
    # Keep original DataFrames for later use
    return X_val, y_val, x_val_df, y_val_df

def load_test_data(logger):
    """Load test data."""
    logger.info(f"Loading test data from {TEST_X_PATH}")
    X_test = pd.read_parquet(TEST_X_PATH)
    y_test_df = pd.read_csv(SKELETON)
    logger.info(f"Test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    return X_test, y_test_df

def load_combined_data(logger):
    """Load and combine all training data sources."""
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
            
            # Free memory
            del x_data, y_data, dataset
            free_memory()
            
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
    free_memory()
    
    return X_combined, y_combined, y_combined_df

def train_model(X_train, y_train, hyperparams, logger):
    """Train an AutoPyTorch model with the given hyperparameters."""
    optimize_metric, walltime_limit, eval_time_limit = hyperparams
    
    model = AutoPyTorchModel(
        optimize_metric=optimize_metric, 
        walltime_limit=walltime_limit,
        eval_time_limit=eval_time_limit
    )
    
    logger.info(f"Model hyperparameters: {model.log_params()}")
    cur_time = datetime.datetime.now()
    logger.info("Training AutoPyTorch ensemble...")
    model.fit(X_train, y_train)
    logger.info(f"Model training completed. Took: {datetime.datetime.now() - cur_time}")
    
    return model

def save_model_and_plot(model, basename, logger):
    """Save the model and generate a performance plot."""
    model_path = os.path.join(LOG_DIR, f"{basename}_model.pkl")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    plot_path = os.path.join(LOG_DIR, f"{basename}_performance.png")
    model.plot_perf_over_time(plot_path)
    logger.info(f"Performance over time plot saved to {plot_path}")

def evaluate_on_validation(model, X_val, y_val, x_val_df, y_val_df, x_train, y_train_df, logger):
    """Evaluate model on validation set using different methods."""
    # Basic prediction and evaluation
    logger.info("Predicting on validation set (transaction level)...")
    y_val_pred = model.predict(X_val)
    report = classification_report(y_val, y_val_pred)
    logger.info("Classification Report on Validation Set (transaction level):\n" + report)
    
    # Free up memory
    del y_val_pred
    free_memory()
    
    # Test different aggregation methods
    aggregation_methods = [
        ("predict_and_aggregate_threshold", predict_and_aggregate_threshold, dict(method='threshold')),
        ("predict_and_aggregate_proba", predict_and_aggregate_proba, {}),
        ("predict_and_aggregate_threshold", predict_and_aggregate_threshold, dict(method='threshold_sum')),
        ("predict_and_aggregate_threshold", predict_and_aggregate_threshold, dict(method='mean'))
    ]
    
    for name, method, params in aggregation_methods:
        logger.info(f"Classification Report **Per Account** on Validation Set:\n{name}({params})")
        accountids_val = method(model, x_val_df, logger=logger, **params)
        accountids_val = accountids_val.set_index('AccountID').reindex(y_val_df['AccountID']).reset_index()
        assert all(accountids_val['AccountID'] == y_val_df['AccountID'])
        report = classification_report(y_val_df['Fraudster'], accountids_val['Fraudster'])
        logger.info(report)
        logger.info('--------------------------------------------------')
        
        # Free memory
        del accountids_val
        free_memory()
    
    # Use ProbaTrain for validation
    logger.info("Classification Report **Per Account** on Validation Set with ProbaTrain")
    proba_train = ProbaTrain(model=model, logger=logger)
    proba_train.fit(x_train, y_train_df)
    accountids_val_proba_train = proba_train.predict(x_val_df, skeleton=y_val_df)
    report = classification_report(y_val_df['Fraudster'], accountids_val_proba_train['Fraudster'])
    logger.info(report)
    logger.info('--------------------------------------------------')
    
    # Free memory
    del proba_train, accountids_val_proba_train
    free_memory()

def predict_on_test(model, basename, X_test, y_test_df, x_train, y_train_df, logger):
    """Generate predictions on test data using different methods."""
    # Get account IDs
    test_ids = X_test["AccountID"]
    
    # Transaction-level predictions
    logger.info("Predicting on test set (transaction level)...")
    y_test_pred = model.predict(X_test)
    out_df = pd.DataFrame({
        "AccountID": test_ids,
        "Fraudster": y_test_pred
    })
    logs_test_path = os.path.join(LOG_DIR, f"{basename}_test_transaction.csv")
    out_df.to_csv(logs_test_path, index=False)
    logger.info(f"Test predictions (transaction level) written to {logs_test_path}")
    
    # Free memory
    del y_test_pred, out_df
    free_memory()

    # Account-level predictions with threshold aggregation
    logger.info("Generating account-level predictions with threshold aggregation...")
    accountids = predict_and_aggregate_threshold(model, X_test, method='threshold', logger=logger)
    accountids = accountids.set_index('AccountID').reindex(y_test_df['AccountID']).reset_index()
    assert all(accountids['AccountID'] == y_test_df['AccountID'])
    logs_test_path = os.path.join(LOG_DIR, f"{basename}_test.csv")
    accountids.to_csv(logs_test_path, index=False)
    logger.info(f"Test account id predictions written to {logs_test_path}")
    
    # Free memory
    del accountids
    free_memory()
    
    # Account-level predictions with ProbaTrain
    logger.info("Generating account-level predictions with ProbaTrain...")
    proba_train = ProbaTrain(model=model, logger=logger)
    proba_train.fit(x_train, y_train_df)
    accountids_proba_train = proba_train.predict(X_test, skeleton=y_test_df)
    logs_test_proba_path = os.path.join(LOG_DIR, f"{basename}_test_proba_train.csv")
    accountids_proba_train.to_csv(logs_test_proba_path, index=False)
    logger.info(f"Test account id predictions with ProbaTrain written to {logs_test_proba_path}")
    
    # Free memory
    del proba_train, accountids_proba_train
    free_memory()

def full_retrain_workflow(hyperparams, basename, logger):
    """Run the full retrain workflow: train on all data and predict on test set."""
    optimize_metric, walltime_limit, eval_time_limit = hyperparams
    
    # Load combined data
    X_combined, y_combined, y_combined_df = load_combined_data(logger)
    log_memory_usage(logger, "After loading combined data:")
    
    # Train model on combined data
    logger.info("Training model on combined data...")
    full_model = train_model(X_combined, y_combined, hyperparams, logger)
    log_memory_usage(logger, "After training full model:")
    
    # Save model and generate performance plot
    save_model_and_plot(full_model, f"{basename}_full", logger)
    
    # Load test data
    X_test, y_test_df = load_test_data(logger)
    log_memory_usage(logger, "After loading test data:")
    
    # Generate predictions
    predict_on_test(full_model, f"{basename}_full", X_test, y_test_df, X_combined, y_combined_df, logger)
    log_memory_usage(logger, "After generating test predictions:")
    
    # Free memory
    del X_combined, y_combined, y_combined_df, full_model, X_test, y_test_df
    free_memory()
    
    logger.info("Full retrain and prediction process complete.")

@click.command()
@click.option('--optimize_metric', default='f1', help='Metric to optimize (e.g., f1, accuracy, roc_auc)')
@click.option('--walltime_limit', default=600, help='Total walltime limit for AutoPyTorch in seconds')
@click.option('--eval_time_limit', default=60, help='Evaluation time limit per model in seconds')
@click.option('--full_retrain', is_flag=True, default=False, help='If set, retrain on all data after evaluation and predict test set')
def main(optimize_metric, walltime_limit, eval_time_limit, full_retrain):
    """
    This script runs AutoPyTorch to train a model ensemble on the training set,
    evaluates it on the validation set (printing a classification report),
    and then uses it to predict fraud labels for the test set.
    
    If --full_retrain flag is set, it will also train a new model on the combined
    training, validation, and kaggle datasets, and use that model to make final 
    predictions on the test set.
    """
    global LOG_BASENAME
    LOG_BASENAME += f"_autoPyTorch_{optimize_metric}_{walltime_limit}s"
    if full_retrain:
        LOG_BASENAME += "_full_retrain"

    logger = setup_logger()
    # log configuration
    logger.info(f"FTSET: {FTSET}")
    logger.info(f"Full Retrain: {full_retrain}")
    logger.info(f"Optimize Metric: {optimize_metric}")
    logger.info(f"Walltime Limit: {walltime_limit} seconds")
    logger.info(f"Evaluation Time Limit: {eval_time_limit} seconds")
    logger.info(f"TRAIN_X_PATH: {TRAIN_X_PATH}")
    logger.info(f"TRAIN_Y_PATH: {TRAIN_Y_PATH}")
    logger.info(f"VAL_X_PATH: {VAL_X_PATH}")
    logger.info(f"VAL_Y_PATH: {VAL_Y_PATH}")
    logger.info(f"TEST_X_PATH: {TEST_X_PATH}")
    logger.info(f"TEST_OUTPUT_PATH: {TEST_OUTPUT_PATH}")
    
    # Track and log initial memory usage
    log_memory_usage(logger, "Initial")
    
    # Hyperparameters
    hyperparams = (optimize_metric, walltime_limit, eval_time_limit)
    
    try:
        # Load training data
        X_train, y_train = load_train_data(logger)
        log_memory_usage(logger, "After loading training data:")
        
        # Keep a copy for ProbaTrain
        x_train = X_train.copy()
        y_train_df = pd.DataFrame({"AccountID": X_train["AccountID"], "Fraudster": y_train})
        
        # Train model
        model = train_model(X_train, y_train, hyperparams, logger)
        log_memory_usage(logger, "After model training:")
        
        # Save model and plot
        save_model_and_plot(model, LOG_BASENAME, logger)
        
        # Free up memory from training data
        del X_train, y_train
        free_memory()
        log_memory_usage(logger, "After freeing training data:")
        
        # Load validation data
        X_val, y_val, x_val_df, y_val_df = load_validation_data(logger)
        log_memory_usage(logger, "After loading validation data:")
        
        # Evaluate on validation set
        evaluate_on_validation(model, X_val, y_val, x_val_df, y_val_df, x_train, y_train_df, logger)
        log_memory_usage(logger, "After validation evaluation:")
        
        # Free validation data
        del X_val, y_val, x_val_df, y_val_df
        free_memory()
        log_memory_usage(logger, "After freeing validation data:")
        
        # Load test data
        X_test, y_test_df = load_test_data(logger)
        log_memory_usage(logger, "After loading test data:")
        
        # Predict on test set
        predict_on_test(model, LOG_BASENAME, X_test, y_test_df, x_train, y_train_df, logger)
        log_memory_usage(logger, "After test prediction:")
        
        # Free test data and model
        del X_test, y_test_df, model
        free_memory()
        log_memory_usage(logger, "After freeing test data and model:")
        
        logger.info("Initial evaluation complete.")
        
        # Full retrain if requested
        if full_retrain:
            logger.info("Starting full retrain process...")
            full_retrain_workflow(hyperparams, LOG_BASENAME, logger)
            
        # Final memory check
        log_memory_usage(logger, "Final")
        
    except Exception as e:
        logger.error(f"Error in main workflow: {str(e)}", exc_info=True)
    
    logger.info("Script execution completed.")

if __name__ == "__main__":
    # example usage:
    # python evaluate_autoPytorch.py --optimize_metric f1 --walltime_limit 600 --eval_time_limit 60
    # python evaluate_autoPytorch.py --optimize_metric accuracy --walltime_limit 1200 --full_retrain
    main()