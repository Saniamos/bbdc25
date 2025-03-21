import pandas as pd
import numpy as np

def predict_and_aggregate(model, x_df, fraudster_percentage=0.13, logger=None):
    logger = logger.info if logger else print
    y_pred = model.predict_proba(x_df)
    out_df = pd.DataFrame({
        "AccountID": x_df["AccountID"],
        "Fraudster": y_pred
    })
    grp = out_df.groupby('AccountID')["Fraudster"].mean()
    
    # Convert Series to numpy array for consistent processing
    account_ids = grp.index.values
    probs = grp.values
    
    # Calculate threshold using numpy's quantile
    # Start with the given percentage and decrease until the selected percentage is below the target
    target_count = int(len(probs) * fraudster_percentage)
    
    # Loop through decreasing percentiles to find the right threshold
    for i in range(int(fraudster_percentage * 100), 0, -1):
        i_percent = i / 100  # Convert to percentage (i.e., 13 -> 0.13)
        threshold = np.quantile(probs, 1 - i_percent)
        if np.sum(probs >= threshold) <= target_count:
            break
    
    # Calculate the percentage of accounts selected
    selected_percentage = np.mean(probs >= threshold) * 100
    logger(f"Threshold: {threshold}, Selected: {selected_percentage:.2f}%")
    
    # Create the final result DataFrame
    fraudster_flags = (probs >= threshold).astype(int)
    aggregated = pd.DataFrame({
        "AccountID": account_ids,
        "Fraudster": fraudster_flags
    })
    
    return aggregated