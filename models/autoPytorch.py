import os
from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.utils.results_visualizer import PlotSettingParams
from models.features.encode import Features


# AutoPyTorch Model wrapper to match the interface expected in evaluate.py
class AutoPyTorchModel:
    # limits are in seconds
    def __init__(self, optimize_metric='f1', walltime_limit=60*30, eval_time_limit=60*2):
        self.optimize_metric = optimize_metric
        self.walltime_limit = walltime_limit
        self.eval_time_limit = eval_time_limit
        self.api = TabularClassificationTask()
        self.fts = Features()
        
    def fit(self, X, y):
        X_train = self.fts._fit_preprocess(X)
        self.api.search(
            X_train=X_train,
            y_train=y,
            optimize_metric=self.optimize_metric,
            total_walltime_limit=self.walltime_limit,
            func_eval_time_limit_secs=self.eval_time_limit,
            # memory_limit=
            refit=False
        )
        return self
    
    def predict(self, X):
        X_pred = self.fts._predict_preprocess(X)
        return self.api.predict(X_pred)
    
    def predict_proba(self, X):
        X_pred = self.fts._predict_preprocess(X)
        return self.api.predict_proba(X_pred)
    
    def refit(self, X, y):
        # """Refit the model to the given data."""
        X_train = self.fts._fit_preprocess(X)
        self.api.refit(X_train=X_train, y_train=y)
        # self.api.refit((X, y), self.optimize_metric, resampling_strategy="no_resampling")
        return self
    
    def score(self, X, y):
        X_score = X.drop('AccountID', axis=1) if 'AccountID' in X.columns else X
        y_pred = self.predict(X_score)
        return self.api.score(y_pred, y)
    
    def save(self, path):
        self.plot_perf_over_time(path.replace('.pkl', '.png'))

        # Extract directory path
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        # Save the model
        self.api.save_models(path)
        return True
        
    def load(self, path):
        self.api.load_models(path)
        return self
    
    def log_params(self):
        return {
            "optimize_metric": self.optimize_metric,
            "walltime_limit": self.walltime_limit,
            "eval_time_limit": self.eval_time_limit
        }
    
    def plot_perf_over_time(self, output_path, show=False):
        params = PlotSettingParams(
            xscale='log',
            xlabel='Runtime',
            ylabel=self.optimize_metric,
            title=f'AutoPyTorch Performance ({self.optimize_metric})',
            figname=output_path,
            savefig_kwargs={'bbox_inches': 'tight'},
            show=show
        )
        
        self.api.plot_perf_over_time(
            metric_name=self.optimize_metric,
            plot_setting_params=params
        )