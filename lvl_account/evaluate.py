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

from ssl_models.dataloader import prepare_dataset, load_val, load_test, load_train

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


@click.command()
# Data parameters
@click.option("--data_version", default="ver05", type=str, help="Data version to use")
@click.option("--pretrained_model_path", required=False, default=None, type=str, 
              help="Path to pre-trained SSL model (only needed for certain model types)")

# Model parameters
@click.option("--model_class", default="simple_cnn", type=str, 
              help="Model class to use (module.ClassName format)")
@click.option("--freeze_bert", is_flag=True, default=True, help="Whether to freeze BERT weights")

@click.option("--continue_from_checkpoint", default=None, type=str, help="Path to a checkpoint to continue training")

# Training parameters
@click.option("--batch_size", default=128, type=int, help="Batch size for training")
@click.option("--num_train_epochs", default=15, type=int, help="Number of training epochs")
@click.option("--val_every_epoch", default=5, type=int, help="Number of training epochs after which to run validation")
@click.option("--learning_rate", default=1e-4, type=float, help="Learning rate")
@click.option("--weight_decay", default=0.01, type=float, help="Weight decay")

# Other parameters
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--num_workers", default=4, type=int, help="Number of workers for data loading")
@click.option("--patience", default=3, type=int, help="Early stopping patience")
@click.option("--dry_run", is_flag=True, help="Perform a dry run (setup but no training)")
@click.option("--skeleton_file", default="~/Repositories/bbdc25/task/professional_skeleton.csv", 
              type=str, help="Path to the skeleton file for test predictions")
def main(model_class, data_version, pretrained_model_path, freeze_bert, continue_from_checkpoint, batch_size, 
         num_train_epochs, val_every_epoch, learning_rate, weight_decay, seed, num_workers, patience, dry_run,
         skeleton_file):
    
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
    
    # Prepare datasets
    logger.info("Preparing datasets...")

    common_args = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
    )
    
    logger.info("Loading training data...")
    train_loader = DataLoader(
            prepare_dataset(data_version, mask=False, log_fn=logger.info, fn=load_train),
            shuffle=True,
            drop_last=True,
            **common_args
        )    
    
    logger.info("Loading validation data...")
    val_loader = DataLoader(
            prepare_dataset(data_version, mask=False, log_fn=logger.info, fn=load_val),
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

    # Apply torch.compile if available (PyTorch 2.0+)
    # This is the preferred way to compile models in Lightning
    if hasattr(torch, 'compile'):
        try:
            logger.info("Compiling model with torch.compile()...")
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Model compiled successfully")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}. Training with uncompiled model.")
        
    # Early stopping - using F1 score as our only primary metric
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min',
    )
    
    # Setup tensorboard logger
    tensorboard_logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, 'logs'),
        name=model_class
    )
    logger.info(f"TensorBoard logs will be saved to {tensorboard_logger.log_dir}")

    # Setup checkpointing - using F1 score as our only primary metric
    # base_pt_name = f'{tensorboard_logger.log_dir}/{model_class}'
    base_pt_name = f'{tensorboard_logger.log_dir.replace(output_dir + "/", "")}/{model_class}'
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=output_dir,
        filename=base_pt_name + '-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min',
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
    


    # Calculate and display validation metrics
    logger.info('---------------------------------------------------')
    logger.info("\nEvaluating model on validation set...")

    val_loader_pred = DataLoader(
        prepare_dataset(data_version, mask=False, log_fn=logger.info, fn=load_val),
        shuffle=False,
        drop_last=False,
        **common_args
    )    
      
    val_predictions = trainer.predict(model, val_loader_pred)

    # Gather predictions and true labels
    val_preds = torch.cat([batch["preds"] for batch in val_predictions]).cpu().numpy().astype(int).flatten()
    val_labels = val_loader_pred.dataset.get_fraud_labels()[:len(val_preds)]  # Ensure same length

    # Print classification report
    report = classification_report(val_labels, val_preds, target_names=["Non-Fraud", "Fraud"])
    logger.info("\nValidation Set Classification Report:\n" + report)



    # Generate and save predictions
    logger.info('---------------------------------------------------')
    logger.info("Generating test predictions...")
    
    logger.info("Loading test data...")
    test_loader = DataLoader(
            prepare_dataset(data_version, mask=False, log_fn=logger.info, fn=load_test),
            shuffle=False,
            drop_last=False,
            **common_args
        )
    
    # Use the PyTorch Lightning predict method which calls our predict_step
    predictions = trainer.predict(model, test_loader)

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

    logger.info(f"Predicted fraudster percentage: {aligned_predictions['Fraudster'].mean()}")
    
    # Save predictions
    predictions_output = os.path.join(LOG_DIR, f"{LOG_BASENAME}_test.csv")
    aligned_predictions.to_csv(predictions_output, index=False)
    logger.info(f"Test predictions saved to {predictions_output}")
    
    logger.info("Training and evaluation complete!")


if __name__ == "__main__":
    # Example usage:
    # python3 evaluate.py --model_class bert --pretrained_model_path=./models/features/ssl/saved_models/transaction_bert/final-model.ckpt
    # python3 evaluate.py --model_class simple_cnn
    # python3 evaluate.py --model_class simple_cnn --batch_size=256 --num_train_epochs=30 --val_every_epoch=5
    main()
