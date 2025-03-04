import pandas as pd

def predict_and_aggregate(model, x_df, method='threshold', fraudster_percentage=0.15, logger=None):
    y_pred = model.predict(x_df)
    out_df = pd.DataFrame({
        "AccountID": x_df["AccountID"],
        "Fraudster": y_pred
    })
    if method == 'mean':
        aggregated = out_df.groupby('AccountID')["Fraudster"].mean().round().astype(int).reset_index()
    else:  # threshold (or threshold_sum if desired)
        if method == 'threshold':
            grp = out_df.groupby('AccountID')["Fraudster"].mean()
        else:  # threshold_sum
            grp = out_df.groupby('AccountID')["Fraudster"].sum()
        threshold = grp.quantile(1 - fraudster_percentage)
        selected_percentage = (grp >= threshold).mean() * 100
        
        # If too many accounts would be selected (e.g., all or nearly all)
        if selected_percentage > 20:  # If more than 20% would be selected
            # Find the threshold that selects closest to but not more than 15%
            sorted_vals = sorted(grp.unique(), reverse=True)
            target_count = int(len(grp) * fraudster_percentage)
            
            for val in sorted_vals:
                if (grp >= val).sum() <= target_count:
                    threshold = val
                else:
                    break
        
        if logger:
            selected_percentage = (grp >= threshold).mean() * 100
            logger.info(f"Threshold: {threshold}, Selected: {selected_percentage:.2f}%")
        aggregated = (grp >= threshold).astype(int).reset_index()
    return aggregated
