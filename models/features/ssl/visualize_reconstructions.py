import os
import click
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from torch.utils.data import DataLoader, Subset
import random
from collections import OrderedDict

from bert import TransactionBERTModel
from dataloader import prepare_datasets, load_val


@click.command()
@click.option("--model_path", required=True, type=str, help="Path to trained model checkpoint")
@click.option("--data_version", default="ver05", type=str, help="Data version to use")
@click.option("--output_dir", default="./visualizations", type=str, help="Directory to save visualizations")
@click.option("--seed", default=42, type=int, help="Random seed")
def main(model_path, data_version, output_dir, seed):
    """Visualize transaction data reconstructions from a trained BERT model."""
    
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load validation dataset
    print(f"Loading validation dataset from version {data_version}...")
    val_dataset = prepare_datasets(data_version, fns=[load_val])[0]
    
    # Directly use account_to_label to find fraudsters and non-fraudsters
    fraud_account_ids = [account_id for account_id, is_fraud in val_dataset.account_to_label.items() if is_fraud > 0.5]
    non_fraud_account_ids = [account_id for account_id, is_fraud in val_dataset.account_to_label.items() if is_fraud <= 0.5]
    
    # Convert account IDs to dataset indices
    account_id_to_idx = {account_id: idx for idx, account_id in enumerate(val_dataset.account_ids)}
    
    print(f"Found {len(fraud_account_ids)} fraud accounts and {len(non_fraud_account_ids)} non-fraud accounts")
    
    # Select 2 from each category
    selected_account_ids = []
    fraud_status = []
    
    # Get 2 fraud accounts (or as many as available)
    if fraud_account_ids:
        selected_fraud = random.sample(fraud_account_ids, min(2, len(fraud_account_ids)))
        selected_account_ids.extend(selected_fraud)
        fraud_status.extend([True] * len(selected_fraud))
    
    # Get 2 non-fraud accounts (or as many as available)
    if non_fraud_account_ids:
        selected_non_fraud = random.sample(non_fraud_account_ids, min(2, len(non_fraud_account_ids)))
        selected_account_ids.extend(selected_non_fraud)
        fraud_status.extend([False] * len(selected_non_fraud))
    
    # Convert to dataset indices
    selected_indices = [account_id_to_idx[account_id] for account_id in selected_account_ids]
    
    # Load model from checkpoint
    print(f"Loading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dim = val_dataset.feature_dim
    
    model = TransactionBERTModel(feature_dim=feature_dim)
    
    # Load state dict from checkpoint file
    if model_path.endswith('.ckpt'):
        # Lightning checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['state_dict']
    else:
        # Regular PyTorch checkpoint
        state_dict = torch.load(model_path, map_location=device)
    
    # Load the state dict directly without fixing prefixes
    # The model was created without compilation, so keys should match
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully. Generating visualizations for {len(selected_indices)} specific accounts...")
    
    # Plot selected examples (2 fraud, 2 non-fraud)
    for i, (idx, is_fraud, account_id) in enumerate(zip(selected_indices, fraud_status, selected_account_ids)):
        masked_seqs, masked_pos, orig_seqs, label = val_dataset[idx]
        account_type = "Fraudster" if is_fraud else "Non-fraudster"
        
        # Get only the first 100 timesteps for clarity in visualization
        max_timesteps = min(100, masked_seqs.shape[0])
        
        # Convert to tensors and move to device
        masked_seqs_tensor = masked_seqs[:max_timesteps].unsqueeze(0).to(device)  # Add batch dimension
        masked_pos_tensor = masked_pos[:max_timesteps].unsqueeze(0).to(device)
        orig_seqs_tensor = orig_seqs[:max_timesteps].to(device)
        
        # Forward pass to get reconstruction
        with torch.no_grad():
            reconstructed_seqs = model(masked_seqs_tensor, masked_pos_tensor).squeeze(0)
            
        # Move tensors back to CPU for plotting
        masked_seqs_np = masked_seqs[:max_timesteps].cpu().numpy()
        masked_pos_np = masked_pos[:max_timesteps].cpu().numpy()
        orig_seqs_np = orig_seqs[:max_timesteps].cpu().numpy()
        reconstructed_seqs_np = reconstructed_seqs.cpu().numpy()
        
        # Create a combined reconstruction where only masked positions use reconstructed values
        combined_recon_np = orig_seqs_np.copy()  # Start with original values
        combined_recon_np[masked_pos_np] = reconstructed_seqs_np[masked_pos_np]  # Replace only masked positions
        
        # Print some statistics to help debug
        print(f"Example {i+1} ({account_type}):")
        print(f"- Original values range: [{orig_seqs_np.min():.4f}, {orig_seqs_np.max():.4f}]")
        print(f"- Reconstructed values range: [{reconstructed_seqs_np.min():.4f}, {reconstructed_seqs_np.max():.4f}]")
        print(f"- Number of masked positions: {masked_pos_np.sum()}")
        
        # Calculate difference between original and combined reconstruction
        # This will show the error only at masked positions
        diff = orig_seqs_np - combined_recon_np
        
        # Create a count of non-zero features per timestep to see actual sequence length
        nonzero_count = np.count_nonzero(orig_seqs_np, axis=1)
        actual_seq_len = np.max(np.where(nonzero_count > 0)[0]) + 1 if np.any(nonzero_count > 0) else max_timesteps
        
        # Trim data to actual sequence length
        masked_seqs_np = masked_seqs_np[:actual_seq_len]
        masked_pos_np = masked_pos_np[:actual_seq_len]
        orig_seqs_np = orig_seqs_np[:actual_seq_len]
        reconstructed_seqs_np = reconstructed_seqs_np[:actual_seq_len]
        combined_recon_np = combined_recon_np[:actual_seq_len]
        diff = diff[:actual_seq_len]
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(4, 1, figsize=(15, 20))
        
        # Find global min and max values for consistent color scaling
        vmin = min(orig_seqs_np.min(), reconstructed_seqs_np.min())
        vmax = max(orig_seqs_np.max(), reconstructed_seqs_np.max())
        
        # Create a diverging colormap for the difference plot
        diff_norm = TwoSlopeNorm(vmin=diff.min(), vcenter=0, vmax=diff.max())
        
        # Plot masked positions overlay
        mask_overlay = np.zeros_like(orig_seqs_np)
        mask_overlay[masked_pos_np] = 1.0
        
        # Plot original data
        sns.heatmap(orig_seqs_np, ax=axes[0], cmap="icefire", vmin=vmin, vmax=vmax)
        axes[0].set_title("Original Transaction Data")
        axes[0].set_xlabel("Feature Dimension")
        axes[0].set_ylabel("Time Step")
        
        # Plot masked input data with masked positions highlighted
        sns.heatmap(masked_seqs_np, ax=axes[1], cmap="icefire", vmin=vmin, vmax=vmax)
        # Overlay a semi-transparent red mask to show masked positions
        # axes[1].imshow(mask_overlay, alpha=0.3, cmap="Reds")
        axes[1].set_title("Masked Transaction Data (Model Input)")
        axes[1].set_xlabel("Feature Dimension")
        axes[1].set_ylabel("Time Step")
        
        # Plot combined reconstruction (original + reconstructed at masked positions)
        sns.heatmap(combined_recon_np, ax=axes[2], cmap="icefire", vmin=vmin, vmax=vmax)
        # axes[2].imshow(mask_overlay, alpha=0.3, cmap="Blues")  # Highlight where reconstruction happened
        axes[2].set_title("Reconstructed Transaction Data (at masked positions)")
        axes[2].set_xlabel("Feature Dimension")
        axes[2].set_ylabel("Time Step")
        
        # Plot difference (error) at masked positions
        sns.heatmap(diff, ax=axes[3], cmap="RdBu_r", norm=diff_norm)
        axes[3].set_title("Difference at Masked Positions (Original - Reconstructed)")
        axes[3].set_xlabel("Feature Dimension")
        axes[3].set_ylabel("Time Step")
        
        # Add account ID to the filename and title for traceability
        account_id_short = account_id[-6:] if len(account_id) > 6 else account_id  # Shortened for display
        plt.suptitle(f"Transaction Reconstruction - {account_type} - Account ID: {account_id_short} (Seq. Length: {actual_seq_len})", 
                    fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save figure
        filename = f"{account_type.lower()}_account_{account_id_short}_{i+1}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150)
        print(f"Saved visualization for {account_type} (Account ID: {account_id_short}) to {os.path.join(output_dir, filename)}")
        plt.close(fig)
    
    print(f"Completed generating visualizations in {output_dir}")

if __name__ == "__main__":
    # Example usage:
    # python3 visualize_reconstructions.py --model_path ./saved_models/transaction_bert/final-model.pt
    main()
