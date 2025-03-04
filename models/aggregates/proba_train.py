import numpy as np
import pandas as pd
import xgboost as xgb

class ProbaTrain:
    def __init__(self, model, fraudster_percentage=0.13, logger=None):
        self.model = model
        self.fraudster_percentage = fraudster_percentage
        self.logger = logger.info if logger else print
        self.account_model = xgb.XGBClassifier(random_state=42)

    def _create_account_features(self, x, skeleton):
        """
        Create account-level features from transaction-level features using vectorized operations.
        """
        # Get fraud probability predictions
        y_pred_fraud = self.model.predict_proba(x)[:, 1]
        
        # Create DataFrame with AccountID and fraud probability
        pred_df = pd.DataFrame({
            "AccountID": x["AccountID"],
            "FraudProb": y_pred_fraud
        })
        
        # First calculate standard aggregations - this is already efficient
        account_features = pred_df.groupby("AccountID")["FraudProb"].agg([
            ("mean", "mean"),
            ("std", "std"),
            ("sum", "sum"),
            ("max", "max"),
            ("min", "min"),
            ("count", "size")
        ]).reset_index()
        
        # Replace for loop with vectorized histogram calculation
        # Group by AccountID and apply histogram function to each group
        def calc_hist(group):
            hist, _ = np.histogram(group, bins=10, range=(0, 1))
            return pd.Series(hist, index=[f'hist_bin_{i}' for i in range(10)])
        
        # Apply histogram function to each group
        hist_df = pred_df.groupby("AccountID")["FraudProb"].apply(calc_hist).unstack().reset_index()
        
        # Merge standard aggregations with histograms
        account_features = pd.merge(account_features, hist_df, on="AccountID", how="left")
        
        # Merge with skeleton to ensure proper order and handle missing accounts
        result = pd.merge(
            skeleton[["AccountID"]], 
            account_features,
            on="AccountID", 
            how="left"
        )
        # use the mean as nan indicator, as std might be nan if too little data is present
        self.logger(f"NaN values in the final result: {result['mean'].isnull().sum()}")

        result = result.fillna(-1)  # Fill missing values with -1
        
        # Convert to numpy array excluding the AccountID column
        output_array = result.iloc[:, 1:].values
        
        self.logger(f"Account features shape: {output_array.shape}")
        return output_array

    def fit(self, x_df, y_df):
        # y_df is now expected to be a DataFrame with AccountID and Fraudster columns
        x_features = self._create_account_features(x_df, skeleton=y_df)
        y_labels = y_df["Fraudster"].values
        
        self.account_model.fit(x_features, y_labels)
        self.logger("Account model trained.")
        return self

    def predict(self, x, skeleton):
        x_features = self._create_account_features(x, skeleton=skeleton)   
        prd = self.account_model.predict_proba(x_features)[:,1] # Get the probability of fraud
        account_ids = skeleton["AccountID"].values
        
        # Calculate threshold using numpy's quantile
        # start with the given percentage and decrease by 0.1 until the selected percentage is below the target
        target_count = int(len(prd) * self.fraudster_percentage)
        for i in range(int(self.fraudster_percentage * 100), 0, -1):
            threshold = np.quantile(prd, 1 - i / 100)
            if np.sum(prd >= threshold) <= target_count:
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
