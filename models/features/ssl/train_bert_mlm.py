import os
import click
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import BertTokenizer, BertConfig

from models.features.ssl.bert import BERTMaskedLM
from models.features.ssl.dataloader import prepare_mlm_datasets

@click.command()
# Data parameters
@click.option("--data_version", default="ver03", type=str, help="Data version to use")
@click.option("--max_seq_length", default=128, type=int, help="Maximum sequence length")
@click.option("--mlm_probability", default=0.15, type=float, help="Masking probability for MLM")

# Model parameters
@click.option("--model_name_or_path", default="bert-base-uncased", type=str, help="Path to pretrained model")
@click.option("--tokenizer_name", default="bert-base-uncased", type=str, help="Tokenizer to use")
@click.option("--output_dir", default="./saved_models/bert_mlm", type=str, help="Output directory for models")

# Training parameters
@click.option("--learning_rate", default=5e-5, type=float, help="Learning rate")
@click.option("--weight_decay", default=0.01, type=float, help="Weight decay")
@click.option("--batch_size", default=32, type=int, help="Batch size for training")
@click.option("--num_train_epochs", default=3, type=int, help="Number of training epochs")
@click.option("--warmup_steps", default=1000, type=int, help="Warmup steps for learning rate scheduler")
@click.option("--max_grad_norm", default=1.0, type=float, help="Max gradient norm")
@click.option("--gradient_accumulation_steps", default=1, type=int, help="Gradient accumulation steps")
@click.option("--fp16", is_flag=True, help="Use 16-bit precision")

# Other parameters
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--num_workers", default=4, type=int, help="Number of workers for data loading")
@click.option("--patience", default=3, type=int, help="Early stopping patience")
@click.option("--do_test", is_flag=True, help="Evaluate on the test set after training")
def main(data_version, max_seq_length, mlm_probability, model_name_or_path, tokenizer_name, 
         output_dir, learning_rate, weight_decay, batch_size, num_train_epochs, warmup_steps,
         max_grad_norm, gradient_accumulation_steps, fp16, seed, num_workers, patience, do_test):
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set seed for reproducibility
    pl.seed_everything(seed)
    
    # Initialize tokenizer
    print(f"Initializing tokenizer from {tokenizer_name}")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    
    # Special tokens for financial data
    additional_special_tokens = [
        "[ACCOUNT]", "[AMOUNT]", "[DATE]", "[ACTION]", 
        "[EXTERNAL_TYPE]", "[FRAUDSTER]"
    ]
    
    # Add special tokens
    special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    # Prepare datasets
    print("Loading and preparing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_mlm_datasets(
        version=data_version, 
        tokenizer=tokenizer,
        max_length=max_seq_length,
        mlm_probability=mlm_probability
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Initialize model
    print(f"Initializing BERT model from {model_name_or_path}")
    model = BERTMaskedLM(
        vocab_size=len(tokenizer),
        pretrained_model_name=model_name_or_path,
        finetune_mlp=True,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps
    )
    
    # Setup checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=output_dir,
        filename='bert-mlm-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min',
        save_weights_only=True
    )
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min'
    )
    
    # Setup tensorboard logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, 'logs'),
        name='bert-mlm'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=num_train_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        gradient_clip_val=max_grad_norm,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        precision=16 if fp16 else 32,
        accumulate_grad_batches=gradient_accumulation_steps
    )
    
    # Train the model
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Save the final model and tokenizer
    final_model_path = os.path.join(output_dir, 'final-model')
    model.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"Training completed. Final model saved to {final_model_path}")
    
    # Test the model
    if do_test:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print("Running evaluation on test set...")
        test_results = trainer.test(model, test_loader)
        print(f"Test results: {test_results}")


if __name__ == "__main__":
    main()