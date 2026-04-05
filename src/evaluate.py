import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def benchmark_metrics(y_true, benchmark_pred) -> dict:
    return regression_metrics(y_true, benchmark_pred)


def build_results_table(dummy_metrics, rf_metrics, benchmark_metrics_dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"Model": "Dummy", **dummy_metrics},
        {"Model": "Random Forest", **rf_metrics},
        {"Model": "AI Benchmark", **benchmark_metrics_dict},
    ])