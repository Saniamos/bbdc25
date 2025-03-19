import os
import click
import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from fraud_classifier import FraudBERTClassifier

# Add parent directory to path to import from SSL module
import sys
sys.path.append('./models/features/ssl') # we need to do this weird thing, as the scripts inside of the task folder should also be able to run on their own
from dataloader import prepare_dataset, load_train_val, load_test
from train_bert_mlm import prep_hpsearch_dataloaders


@click.command()
# Data parameters
@click.option("--data_version", default="ver05", type=str, help="Data version to use")
@click.option("--pretrained_model_path", required=True, type=str, help="Path to pre-trained SSL model")

# Model parameters
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
def main(data_version, pretrained_model_path, output_dir, freeze_bert, continue_from_checkpoint, batch_size, 
         num_train_epochs, val_every_epoch, learning_rate, weight_decay, seed, num_workers, patience, dry_run,
         skeleton_file, predictions_output):
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set seed for reproducibility
    pl.seed_everything(seed)
    
    # Prepare datasets
    train_loader, val_loader, feature_dim = prep_hpsearch_dataloaders(data_version, seed, batch_size, num_workers, load_fn=load_train_val)
    
    test_dataset = prepare_dataset(data_version, load_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Check if pretrained model exists
    if not os.path.exists(pretrained_model_path):
        raise ValueError(f"Pre-trained model not found at {pretrained_model_path}")
    
    # Initialize model
    print(f"Initializing Fraud Classifier with pre-trained TransactionBERT weights")
    model = FraudBERTClassifier(
        feature_dim=feature_dim,
        pretrained_model_path=pretrained_model_path,
        freeze_bert=freeze_bert,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    
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
    logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, 'logs'),
        name='fraud-classifier'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=num_train_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        precision='16-mixed',
        fast_dev_run=dry_run,
        check_val_every_n_epoch=val_every_epoch,
        benchmark=True,  # Optimize CUDA operations
    )
    
    # Train the model
    print("Starting training...")
    torch.set_float32_matmul_precision('high')
    trainer.fit(model, train_loader, val_loader, **({'ckpt_path': continue_from_checkpoint} if continue_from_checkpoint else {}))

    # Save the final model
    final_model_path = os.path.join(output_dir, 'final-fraud-classifier.pt')
    if os.path.exists(final_model_path):
        os.remove(final_model_path)
    trainer.save_checkpoint(final_model_path)
    
    print(f"Training completed. Final model saved to {final_model_path}")
    
    # Generate and save predictions
    print("Generating test predictions...")
    
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
    print(f"Test predictions saved to {predictions_output}")
    


if __name__ == "__main__":
    # Example usage:
    # python3 train_fraud_classifier.py --pretrained_model_path=./models/features/ssl/saved_models/transaction_bert/final-model.ckpt
    main()
