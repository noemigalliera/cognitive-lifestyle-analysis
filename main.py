from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data_loader import load_data, split_features_target
from src.preprocess import build_preprocessor
from src.train import build_dummy_model, build_random_forest_model
from src.evaluate import regression_metrics, benchmark_metrics, build_results_table
from src.utils import get_feature_importance


def main():
    data_path = Path("data/raw/human_cognitive_performance.csv")

    df = load_data(data_path)
    X, y, benchmark = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(X)

    dummy_model = build_dummy_model(preprocessor)
    rf_model = build_random_forest_model(preprocessor)

    dummy_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    y_pred_dummy = dummy_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)
    benchmark_pred = benchmark.loc[X_test.index]

    dummy_scores = regression_metrics(y_test, y_pred_dummy)
    rf_scores = regression_metrics(y_test, y_pred_rf)
    benchmark_scores = benchmark_metrics(y_test, benchmark_pred)

    results = build_results_table(dummy_scores, rf_scores, benchmark_scores)

    print("\nModel comparison:")
    print(results.to_string(index=False))

    print("\nTop 10 feature importances:")
    print(get_feature_importance(rf_model).head(10).to_string())


if __name__ == "__main__":
    main()