from pathlib import Path
import pandas as pd

def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)


def split_features_target(df: pd.DataFrame):
    """Split dataset into features, target, and benchmark."""
    X = df.drop(columns=["User_ID", "Cognitive_Score", "AI_Predicted_Score"])
    y = df["Cognitive_Score"]
    benchmark = df["AI_Predicted_Score"]
    return X, y, benchmark