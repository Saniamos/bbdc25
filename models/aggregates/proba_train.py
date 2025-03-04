import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class ProbaTrain:
    def __init__(self, model, fraudster_percentage=0.13, logger=None):
        self.model = model
        self.fraudster_percentage = fraudster_percentage
        self.logger = logger.info if logger else print
        self.account_model = RandomForestClassifier(random_state=42)

    def _create_account_features(self, x, skeleton):
        """
        Create account-level features from transaction-level features using vectorized operations.
        """
        # Get fraud probability predictions
        y_pred_fraud = self.model.predict_proba(x)[:, 0]
        
        # Create DataFrame with AccountID and fraud probability
        pred_df = pd.DataFrame({
            "AccountID": x["AccountID"],
            "FraudProb": y_pred_fraud
        })
        
        # Calculate all features at once using pandas aggregation
        account_features = pred_df.groupby("AccountID")["FraudProb"].agg([
            ("mean", "mean"),
            ("std", "std"),
            ("sum", "sum"),
            ("max", "max"),
            ("min", "min"),
            ("count", "size")
        ]).reset_index()
        
        # Merge with skeleton to ensure proper order and handle missing accounts
        # Left join to keep all skeleton account IDs
        result = pd.merge(
            skeleton[["AccountID"]].drop_duplicates(), 
            account_features,
            on="AccountID", 
            how="left"
        ).fillna(-1)  # Fill NaN values with 0 for accounts with no transactions
        
        # Convert to numpy array excluding the AccountID column
        output_array = result.iloc[:, 1:].values
        
        return output_array

    def fit(self, x_df, y_df):
        # y_df is now expected to be a DataFrame with AccountID and Fraudster columns
        x_features = self._create_account_features(x_df, skeleton=y_df)
        y_labels = y_df["Fraudster"].values
        
        self.account_model.fit(x_features, y_labels)
        return self

    def predict(self, x, skeleton):
        x_features = self._create_account_features(x, skeleton=skeleton)   
        prd = self.account_model.predict_proba(x_features)[:,1] # Get the probability of fraud
        account_ids = skeleton["AccountID"].values
        
        # Calculate threshold using numpy's quantile
        threshold = np.quantile(prd, 1 - self.fraudster_percentage)
        selected_percentage = np.mean(prd >= threshold) * 100
        
        # If too many accounts would be selected (e.g., all or nearly all)
        if selected_percentage > 20:  # If more than 20% would be selected
            # Find the threshold that selects closest to but not more than the target
            sorted_vals = np.sort(np.unique(prd))[::-1]  # Sort in descending order
            target_count = int(len(prd) * self.fraudster_percentage)
            
            for val in sorted_vals:
                if np.sum(prd >= val) <= target_count:
                    threshold = val
                else:
                    break
        
        selected_percentage = np.mean(prd >= threshold) * 100
        self.logger(f"Threshold: {threshold}, Selected: {selected_percentage:.2f}%")
        
        # Create the final result DataFrame
        fraudster_flags = (prd >= threshold).astype(int)
        aggregated = pd.DataFrame({
            "AccountID": account_ids,
            "Fraudster": fraudster_flags
        })
        
        return aggregated
