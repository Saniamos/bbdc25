import os
import click
import torch
import pandas as pd
import importlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from sklearn.metrics import classification_report
import logging
import datetime

from ssl.dataloader import prepare_dataset, load_train_val, load_test
from ssl.train_bert_mlm import prep_hpsearch_dataloaders

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


def import_model_class(model_class_path, logger):
    """
    Dynamically import a model class from a string path.
    e.g. 'fraud_classifier.Classifer' or 'models.transformer.TransformerClassifier'
    """
    try:
        module_path, class_name = model_class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        return model_class
    except (ImportError, AttributeError) as e:
        logger.error(f"Could not import model class '{model_class_path}': {e}")
        raise ImportError(f"Could not import model class '{model_class_path}': {e}")


@click.command()
# Data parameters
@click.option("--data_version", default="ver05", type=str, help="Data version to use")
@click.option("--pretrained_model_path", required=False, default=None, type=str, 
              help="Path to pre-trained SSL model (only needed for certain model types)")

# Model parameters
@click.option("--model_class", default="fraud_classifier.Classifer", type=str, 
              help="Model class to use (module.ClassName format)")
@click.option("--output_dir", default="./saved_models/fraud_classifier", type=str, help="Output directory for models")
@click.option("--freeze_bert", is_flag=True, default=True, help="Whether to freeze BERT weights")

@click.option("--continue_from_checkpoint", default=None, type=str, help="Path to a checkpoint to continue training")

# Training parameters
@click.option("--batch_size", default=128, type=int, help="Batch size for training")
@click.option("--num_train_epochs", default=10, type=int, help="Number of training epochs")
@click.option("--val_every_epoch", default=3, type=int, help="Number of training epochs after which to run validation")
@click.option("--learning_rate", default=1e-4, type=float, help="Learning rate")
@click.option("--weight_decay", default=0.01, type=float, help="Weight decay")

# Other parameters
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--num_workers", default=5, type=int, help="Number of workers for data loading")
@click.option("--patience", default=3, type=int, help="Early stopping patience")
@click.option("--dry_run", is_flag=True, help="Perform a dry run (setup but no training)")
@click.option("--skeleton_file", default="~/Repositories/bbdc25/task/professional_skeleton.csv", 
              type=str, help="Path to the skeleton file for test predictions")
@click.option("--predictions_output", default=None, type=str, 
              help="Path to save test predictions (defaults to output_dir/predictions.csv)")
def main(model_class, data_version, pretrained_model_path, output_dir, freeze_bert, continue_from_checkpoint, batch_size, 
         num_train_epochs, val_every_epoch, learning_rate, weight_decay, seed, num_workers, patience, dry_run,
         skeleton_file, predictions_output):
    
    # Setup logging
    logger = setup_logger(model_class)
    
    # Log all configuration options
    logger.info(f"Configuration: data_version={data_version}, model_class={model_class}")
    logger.info(f"pretrained_model_path={pretrained_model_path}, output_dir={output_dir}")
    logger.info(f"freeze_bert={freeze_bert}, batch_size={batch_size}, epochs={num_train_epochs}")
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
    train_loader, val_loader, feature_dim = prep_hpsearch_dataloaders(data_version, seed, batch_size, num_workers, load_fn=load_train_val)
    
    test_dataset = prepare_dataset(data_version, load_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
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
    
    # Setup checkpointing - using F1 score as our only primary metric
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=output_dir,
        filename='fraud-bert-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min',  # F1 score should be maximized
        save_weights_only=True
    )
    
    # Early stopping - using F1 score as our only primary metric
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min',  # F1 score should be maximized
    )
    
    # Setup tensorboard logger
    tensorboard_logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, 'logs'),
        name='fraud-classifier'
    )
    logger.info(f"TensorBoard logs will be saved to {os.path.join(output_dir, 'logs')}")
    
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

    # Print model summary before training
    model_summary = ModelSummary(model, max_depth=2)
    logger.info("Model Summary:\n" + str(model_summary))
    
    # Train the model
    logger.info("Starting training...")
    torch.set_float32_matmul_precision('high')
    trainer.fit(model, train_loader, val_loader, **({'ckpt_path': continue_from_checkpoint} if continue_from_checkpoint else {}))

    # Save the final model
    final_model_path = os.path.join(output_dir, 'final-fraud-classifier.pt')
    if os.path.exists(final_model_path):
        os.remove(final_model_path)
    trainer.save_checkpoint(final_model_path)
    
    logger.info(f"Training completed. Final model saved to {final_model_path}")
    

    # Calculate and display validation metrics
    logger.info("\nEvaluating model on validation set...")
    val_predictions = trainer.predict(model, val_loader)

    # Gather predictions and true labels
    val_probs = torch.cat([batch["probs"] for batch in val_predictions]).cpu().numpy()
    val_preds = (val_probs > 0.5).astype(int).flatten()
    val_labels = []

    # Extract validation labels
    for batch in val_loader:
        _, _, _, y = batch
        val_labels.extend(y.cpu().numpy().flatten())

    val_labels = val_labels[:len(val_preds)]  # Ensure same length

    # Print classification report
    report = classification_report(val_labels, val_preds, target_names=["Non-Fraud", "Fraud"])
    logger.info("\nValidation Set Classification Report:\n" + report)


    # Generate and save predictions
    logger.info("Generating test predictions...")
    
    # Use the PyTorch Lightning predict method which calls our predict_step
    predictions = trainer.predict(model, test_loader)
    
    # Concatenate predictions from all batches
    all_probs = torch.cat([batch["probs"] for batch in predictions]).cpu().numpy()
    binary_preds = (all_probs > 0.5).astype(int).flatten()
    
    # Extract account IDs
    test_account_ids = []
    dataset_size = len(test_dataset)
    
    # Process batches to get account IDs
    for i, (indices_start) in enumerate(range(0, dataset_size, batch_size)):
        indices_end = min(indices_start + batch_size, dataset_size)
        batch_indices = range(indices_start, indices_end)
        batch_account_ids = [test_dataset.account_ids[idx] for idx in batch_indices]
        test_account_ids.extend(batch_account_ids)
    
    # Create DataFrame with predictions
    predictions_df = pd.DataFrame({
        "AccountID": test_account_ids[:len(binary_preds)],
        "Fraudster": binary_preds
    })
    
    # Ensure predictions are in the same order as the skeleton file
    skeleton_df = pd.read_csv(skeleton_file)
    aligned_predictions = pd.merge(
        skeleton_df[["AccountID"]], 
        predictions_df, 
        on="AccountID", 
        how="left"
    ).fillna(0)
    
    # Set output path
    if predictions_output is None:
        predictions_output = os.path.join(output_dir, "predictions.csv")
    
    # Save predictions
    aligned_predictions.to_csv(predictions_output, index=False)
    logger.info(f"Test predictions saved to {predictions_output}")
    
    logger.info("Training and evaluation complete!")


if __name__ == "__main__":
    # Example usage:
    # python3 train_fraud_classifier.py --pretrained_model_path=./models/features/ssl/saved_models/transaction_bert/final-model.ckpt
    # python3 train_fraud_classifier.py --model_class fraud_classifier_simple.Classifier
    main()
