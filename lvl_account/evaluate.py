import os
import click
import torch
import pandas as pd
import importlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report
import logging
import datetime
import numpy as np
import gc  # For garbage collection

from ssl_models.dataloader import prepare_dataset, load_val, load_test, load_train, N_FRAUDSTERS

# Global constants for logging
LOG_DIR = "logs"
LOG_BASENAME = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

def setup_logger(model_class):
    """Setup and configure a logger that outputs to both console and file"""
    global LOG_BASENAME
    LOG_BASENAME += f"_{model_class.split('.')[-1]}"
    
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"{LOG_BASENAME}.txt")
    
    logger = logging.getLogger("train_fraud_classifier_logger")
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create console handler and set level to info
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    # Add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    logger.info(f"Logging to file: {log_file}")
    return logger


def import_model_class(python_file_base, logger):
    """
    Dynamically import a model class from a string path.
    e.g. 'fraud_classifier.Classifer' or 'models.transformer.TransformerClassifier'
    """
    try:
        module_path = 'models.' + python_file_base
        class_name = 'Classifier'
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        return model_class
    except (ImportError, AttributeError) as e:
        logger.error(f"Could not import model class '{python_file_base}': {e}")
        raise ImportError(f"Could not import model class '{python_file_base}': {e}")


def train_model(logger, model_class, data_version, precompute, pretrained_model_path, freeze_bert, 
               continue_from_checkpoint, num_train_epochs, val_every_epoch, 
               learning_rate, weight_decay, patience, dry_run, output_dir, common_args, comp):
    """Train the model and save checkpoints"""
    
    logger.info("Preparing datasets for training...")
    
    
    logger.info("Loading training data...")
    train_dataset = prepare_dataset(data_version, mask=False, precompute=precompute, log_fn=logger.info, fn=load_train)
    train_loader = DataLoader(
        train_dataset, 
        shuffle=True,
        drop_last=True,
        **common_args
    )    
    
    logger.info("Loading validation data...")
    val_dataset = prepare_dataset(data_version, mask=False, precompute=precompute, log_fn=logger.info, fn=load_val)
    val_loader = DataLoader(
        val_dataset, 
        shuffle=False,
        drop_last=True,
        **common_args
    )  
    
    feature_dim = train_loader.dataset.feature_dim
    logger.info(f"Data loaders prepared. Feature dimension: {feature_dim}")
    
    # Check if pretrained model exists (only if path is provided)
    if pretrained_model_path is not None:
        if not os.path.exists(pretrained_model_path):
            logger.error(f"Pre-trained model not found at {pretrained_model_path}")
            raise ValueError(f"Pre-trained model not found at {pretrained_model_path}")
        logger.info(f"Using pre-trained model from {pretrained_model_path}")
    else:
        logger.info("No pre-trained model provided, model will be initialized from scratch")

    # Dynamically import the model class
    try:
        ModelClass = import_model_class(model_class, logger)
        logger.info(f"Successfully imported model class: {model_class}")
    except ImportError as e:
        logger.error(f"Error importing model class: {e}")
        import sys
        sys.exit(1)

    # Initialize model with optional pretrained path
    logger.info(f"Initializing {ModelClass.__name__}")
    model_kwargs = {
        "feature_dim": feature_dim,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay
    }

    # Only add pretrained_model_path and freeze_bert if pretrained model is provided
    if pretrained_model_path is not None:
        model_kwargs["pretrained_model_path"] = pretrained_model_path
        model_kwargs["freeze_bert"] = freeze_bert

    # Initialize the model with appropriate arguments
    model = ModelClass(**model_kwargs)

    # # Apply torch.compile if available (PyTorch 2.0+)
    if hasattr(torch, 'compile') and comp:
        try:
            logger.info("Compiling model with torch.compile()...")
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Model compiled successfully")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}. Training with uncompiled model.")
        
    # Early stopping - using F1 score as our only primary metric
    early_stop_callback = EarlyStopping(
        monitor='val_fraud_f1', # range [0, 1]
        patience=patience,
        min_delta=0,
        mode='max',
    )
    
    # Setup tensorboard logger
    tensorboard_logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, 'logs'),
        name=model_class
    )
    logger.info(f"TensorBoard logs will be saved to {tensorboard_logger.log_dir}")

    # Setup checkpointing - using F1 score as our only primary metric
    base_pt_name = f'{tensorboard_logger.log_dir.replace(output_dir + "/", "")}/{model_class}'
    checkpoint_callback = ModelCheckpoint(
        monitor='val_fraud_f1',
        dirpath=output_dir,
        filename=base_pt_name + '-{epoch:02d}-{val_fraud_f1:.4f}',
        save_top_k=3,
        mode='max',
        save_weights_only=True
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=num_train_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=tensorboard_logger,
        precision='16-mixed',
        fast_dev_run=dry_run,
        check_val_every_n_epoch=val_every_epoch,
        benchmark=True,  # Optimize CUDA operations
    )
    
    device_info = "GPU" if torch.cuda.is_available() else "CPU"
    logger.info(f"Using {device_info} for training")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Train the model
    logger.info("Starting training...")
    torch.set_float32_matmul_precision('high')
    trainer.fit(model, train_loader, val_loader, **({'ckpt_path': continue_from_checkpoint} if continue_from_checkpoint else {}))

    # Save the final model
    final_model_path = os.path.join(output_dir, f'{base_pt_name}-final.ckpt')
    if os.path.exists(final_model_path):
        os.remove(final_model_path)
    trainer.save_checkpoint(final_model_path)
    
    logger.info(f"Training completed. Final model saved to {final_model_path}")
    
    # Free up memory
    del train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, trainer, final_model_path, train_dataset, val_dataset

def probs_to_preds(probs, top_n):
    preds = np.zeros_like(probs)
    preds[np.argsort(-probs)[:top_n]] = 1
    return preds

def evaluate_on_validation(logger, trainer, model, common_args, val_dataset):
    """Evaluate the model on the validation set and report metrics"""
    
    logger.info('---------------------------------------------------')
    logger.info("\nEvaluating model on validation set...")

    val_loader_pred = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common_args
    )
    
    n = 1
    if hasattr(model, 'reset_account_pred_state'):
        model.reset_account_pred_state()
        n = 4
    for i in range(n):
        logger.info(f"=== Pass {i} ===================")
        # Use the PyTorch Lightning predict method which calls our predict_step
        val_predictions = trainer.predict(model, val_loader_pred, ckpt_path="best")

        # Gather predictions and true labels
        val_preds = torch.cat([batch["preds"] for batch in val_predictions]).cpu().numpy().astype(int).flatten()
        val_labels = val_loader_pred.dataset.get_fraud_labels()[:len(val_preds)]  # Ensure same length

        # Print classification report
        report = classification_report(val_labels, val_preds, target_names=["Non-Fraud", "Fraud"])
        logger.info("\nValidation Set Classification Report:\n" + report)

    val_probs = torch.cat([batch["probs"] for batch in val_predictions]).cpu().numpy().astype(int).flatten()
    n = N_FRAUDSTERS['val']
    val_probs_prebs = probs_to_preds(val_probs, n)
    report = classification_report(val_labels, val_probs_prebs, target_names=["Non-Fraud", "Fraud"])
    logger.info(f"\nValidation Set Classification Report (top {n}):\n" + report)

    # Save predictions   
    account_ids = val_loader_pred.dataset.get_account_ids()[:len(val_preds)]  # Ensure same length
    predictions_df = pd.DataFrame({
        "AccountID": account_ids,
        "Fraudster": val_preds.astype(int)
    })
    predictions_df['AccountID'] = predictions_df['AccountID'].str.split('yh').str[0]       
    predictions_output = os.path.join(LOG_DIR, f"{LOG_BASENAME}_val.csv")
    predictions_df.to_csv(predictions_output, index=False)
    logger.info(f"Predicted fraudster count: {predictions_df['Fraudster'].sum()} / {N_FRAUDSTERS['val']}")
    logger.info(f"Test predictions saved to {predictions_output}")

    # Free up memory
    del val_loader_pred, val_predictions
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def generate_train_predictions(logger, trainer, model, common_args, train_dataset):
    # Generate predictions for the training set similar to validation and test
    logger.info('---------------------------------------------------')
    logger.info("Generating train predictions...")
    
    train_loader_pred = DataLoader(
        train_dataset,
        shuffle=False,
        drop_last=False,
        **common_args
    )
    
    n = 1
    if hasattr(model, 'reset_account_pred_state'):
        model.reset_account_pred_state()
        n = 4
    for i in range(n):
        logger.info(f"=== Pass {i} ===================")
        train_predictions = trainer.predict(model, train_loader_pred, ckpt_path="best")
    
    preds = torch.cat([batch["preds"] for batch in train_predictions]).cpu().numpy().astype(int).flatten()
    account_ids = train_loader_pred.dataset.get_account_ids()[:len(preds)]
    
    predictions_df = pd.DataFrame({
        "AccountID": account_ids,
        "Fraudster": preds.astype(int)
    })
    predictions_df['AccountID'] = predictions_df['AccountID'].str.split('yh').str[0]
    
    predictions_output = os.path.join(LOG_DIR, f"{LOG_BASENAME}_train.csv")
    predictions_df.to_csv(predictions_output, index=False)
    logger.info(f"Train predictions saved to {predictions_output}")
    
    # Free up memory
    del train_loader_pred, train_predictions
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return predictions_output

def generate_test_predictions(logger, trainer, model, data_version, precompute, common_args, skeleton_file):
    """Generate predictions on the test set and save to CSV"""
    
    logger.info('---------------------------------------------------')
    logger.info("Generating test predictions...")
    
    logger.info("Loading test data...")
    test_loader = DataLoader(
        prepare_dataset(data_version, mask=False, precompute=precompute, log_fn=logger.info, fn=load_test),
        shuffle=False,
        drop_last=False,
        **common_args
    )

    n = 1
    if hasattr(model, 'reset_account_pred_state'):
        model.reset_account_pred_state()
        n = 3
    for i in range(n):
        logger.info(f"=== Pass {i} ===================")
        # Use the PyTorch Lightning predict method which calls our predict_step
        predictions = trainer.predict(model, test_loader, ckpt_path="best")

    # Gather predictions and account ids labels
    preds = torch.cat([batch["preds"] for batch in predictions]).cpu().numpy().astype(int).flatten()
    account_ids = test_loader.dataset.get_account_ids()[:len(preds)]  # Ensure same length

    # Create DataFrame with predictions
    predictions_df = pd.DataFrame({
        "AccountID": account_ids,
        "Fraudster": preds.astype(int)
    })
    predictions_df['AccountID'] = predictions_df['AccountID'].str.split('yh').str[0]       
    
    # Ensure predictions are in the same order as the skeleton file
    skeleton_df = pd.read_csv(skeleton_file)
    aligned_predictions = pd.merge(
        skeleton_df[["AccountID"]], 
        predictions_df, 
        on="AccountID", 
        how="left"
    ).fillna(0)

    logger.info(f"Predicted fraudster count: {aligned_predictions['Fraudster'].sum()} / {N_FRAUDSTERS['test']}")
    
    # Save predictions
    predictions_output = os.path.join(LOG_DIR, f"{LOG_BASENAME}_test.csv")
    aligned_predictions.to_csv(predictions_output, index=False)
    logger.info(f"Test predictions saved to {predictions_output}")
    
    # Free up memory
    del test_loader, predictions
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return predictions_output


@click.command()
# Data parameters
@click.option("--data_version", default="ver05", type=str, help="Data version to use")
@click.option("--precompute", default=False, is_flag=True, help="If True, precompute features and save to disk")
@click.option("--pretrained_model_path", required=False, default=None, type=str, 
              help="Path to pre-trained SSL model (only needed for certain model types)")

# Model parameters
@click.option("--model_class", default="rec_cnn", type=str, 
              help="Model class to use (module.ClassName format)")
@click.option("--freeze_bert", is_flag=True, default=True, help="Whether to freeze BERT weights")

@click.option("--continue_from_checkpoint", default=None, type=str, help="Path to a checkpoint to continue training")

# Training parameters
@click.option("--batch_size", default=128, type=int, help="Batch size for training")
@click.option("--num_train_epochs", default=1, type=int, help="Number of training epochs")
@click.option("--val_every_epoch", default=3, type=int, help="Number of training epochs after which to run validation")
@click.option("--learning_rate", default=1e-4, type=float, help="Learning rate")
@click.option("--weight_decay", default=0.01, type=float, help="Weight decay")

# Other parameters
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--num_workers", default=0, type=int, help="Number of workers for data loading")
@click.option("--patience", default=3, type=int, help="Early stopping patience")
@click.option("--dry_run", is_flag=True, help="Perform a dry run (setup but no training)")
@click.option("--skeleton_file", default="~/Repositories/bbdc25/task/professional_skeleton.csv", 
              type=str, help="Path to the skeleton file for test predictions")
@click.option("--skip_train", is_flag=True, help="Skip training and use existing model")
@click.option("--skip_validation", is_flag=True, help="Skip validation evaluation")
@click.option("--skip_test", is_flag=True, help="Skip test prediction")
@click.option("--comp", is_flag=True, default=False, help="Compile model if available (PyTorch 2.0+)")
def main(model_class, data_version, precompute, pretrained_model_path, freeze_bert, continue_from_checkpoint, batch_size, 
         num_train_epochs, val_every_epoch, learning_rate, weight_decay, seed, num_workers, patience, dry_run,
         skeleton_file, skip_train, skip_validation, skip_test, comp):
    
    # Setup logging
    logger = setup_logger(model_class)
    
    # Log all configuration options
    logger.info(f"Configuration: data_version={data_version}, model_class={model_class}")
    output_dir = os.path.join("saved_models", model_class)
    logger.info(f"pretrained_model_path={pretrained_model_path}, output_dir={output_dir}")
    logger.info(f"freeze_bert={freeze_bert}, batch_size={batch_size}, epochs={num_train_epochs}")
    val_every_epoch = min(val_every_epoch, num_train_epochs)
    logger.info(f"val_every_epoch={val_every_epoch}, learning_rate={learning_rate}, weight_decay={weight_decay}")
    logger.info(f"seed={seed}, num_workers={num_workers}, patience={patience}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Set seed for reproducibility
    pl.seed_everything(seed)
    logger.info(f"Set random seed to {seed}")
    
    # Setup common args
    common_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    if not skip_train:
        # Train the model
        model, trainer, final_model_path, train_dataset, val_dataset = train_model(
            logger, model_class, data_version, precompute, pretrained_model_path, freeze_bert, 
            continue_from_checkpoint, num_train_epochs, val_every_epoch, 
            learning_rate, weight_decay, patience, dry_run, output_dir, common_args, comp
        )
    else:
        # Load existing model
        logger.info("Skipping training and loading existing model...")
        final_model_path = continue_from_checkpoint
        if final_model_path is None:
            logger.error("Must provide --continue_from_checkpoint when using --skip_train")
            return
        
        
        # Import model class
        ModelClass = import_model_class(model_class, logger)
        
        # Initialize trainer
        trainer = pl.Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            precision='16-mixed',
        )
        
        # Load the model
        model = ModelClass.load_from_checkpoint(final_model_path)
        logger.info(f"Loaded model from {final_model_path}")
        train_dataset.delete_cache()
    
    if not skip_validation:            
        # Evaluate on validation set
        evaluate_on_validation(logger, trainer, model, common_args, val_dataset)
        val_dataset.delete_cache()

    # Add train predictions output if training dataset is available.
    if not skip_train:
        generate_train_predictions(logger, trainer, model, common_args, train_dataset)
    else:
        logger.info("Skipping train predictions because train dataset is not available.")
    
    if not skip_test:
        # Generate test predictions
        generate_test_predictions(logger, trainer, model, data_version, precompute, common_args, skeleton_file)
    
    logger.info("Process complete!")


if __name__ == "__main__":
    # Example usage:
    # python3 evaluate.py --model_class bert --pretrained_model_path=./models/features/ssl/saved_models/transaction_bert/final-model.ckpt
    # python3 evaluate.py --model_class simple_cnn
    # python3 evaluate.py --model_class simple_cnn --batch_size=256 --num_train_epochs=30 --val_every_epoch=5
    # python3 evaluate.py --skip_train --continue_from_checkpoint=./saved_models/simple_cnn/simple_cnn-final.ckpt
    main()