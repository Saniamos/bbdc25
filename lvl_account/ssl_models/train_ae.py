import os
import click
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from ae import AutoEncoder
from dataloader import load_train, prepare_dataset

@click.command()
@click.option("--data_version", default="ver05", type=str, help="Data version for training")
@click.option("--batch_size", default=221, type=int, help="Batch size for training")
@click.option("--num_epochs", default=10, type=int, help="Number of training epochs")
@click.option("--learning_rate", default=1e-3, type=float, help="Learning rate for training")
@click.option("--num_workers", default=0, type=int, help="Number of DataLoader workers")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--early_stop_patience", default=3, type=int, help="Early stopping patience")
@click.option("--output_dir", default="./saved_models/autoencoder", type=str, help="Output directory for model and logs")
@click.option("--precompute", is_flag=True, default=False, help="Precompute features for faster training")
@click.option("--comp", is_flag=True, default=False, help="Precompute features for faster training")
def main(data_version, batch_size, num_epochs, learning_rate, num_workers,
         seed, early_stop_patience, output_dir, precompute, comp):
    # Create output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed for reproducibility.
    pl.seed_everything(seed)
    
    # Prepare dataset and DataLoader.
    dataset = prepare_dataset(data_version, load_train, mask=False, log_fn=print, max_seq_len=2048, precompute=precompute)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    feature_dim = dataset.feature_dim
    
    # Initialize model.
    model = AutoEncoder(feature_dim=feature_dim, learning_rate=learning_rate)
    
    # Optionally compile model if available (PyTorch 2.0+)
    if hasattr(torch, 'compile') and comp:
        try:
            print("Compiling model with torch.compile()...")
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled successfully")
        except Exception as e:
            print(f"Compilation failed: {e}. Training with uncompiled model.")
    
    # Set up callbacks.
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath=output_dir,
        filename='autoencoder-{epoch:02d}-{train_loss:.4f}',
        save_top_k=1,
        mode='min'
    )
    early_stop_callback = EarlyStopping(monitor='train_loss', patience=early_stop_patience, mode='min')
    
    # Setup TensorBoard logger.
    logger = TensorBoardLogger(save_dir=os.path.join(output_dir, 'logs'), name='autoencoder')
    
    # Initialize trainer.
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=16,  # added precision for fp16 training
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        benchmark=True,
        gradient_clip_val=1.0  # added gradient clipping to mitigate NaNs
    )
    
    # Set float32 matmul precision.
    torch.set_float32_matmul_precision('high')
    
    # Train the model.
    trainer.fit(model, train_dataloaders=dataloader)
    
    # Save final model checkpoint.
    final_path = os.path.join(output_dir, "final-model.ckpt")
    if os.path.exists(final_path):
        os.remove(final_path)
    trainer.save_checkpoint(final_path)
    print(f"Training completed. Final model saved to {final_path}")

if __name__ == "__main__":
    # example usage
    # python3 train_ae.py --data_version ver05 --num_epochs 10 --precompute
    main()
