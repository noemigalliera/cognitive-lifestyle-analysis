from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


def build_dummy_model(preprocessor):
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", DummyRegressor(strategy="mean"))
    ])


def build_random_forest_model(preprocessor):
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ))
    ])