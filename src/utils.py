import pandas as pd


def get_feature_importance(model) -> pd.Series:
    """Extract feature importance from trained random forest pipeline."""
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importances = model.named_steps["model"].feature_importances_
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)