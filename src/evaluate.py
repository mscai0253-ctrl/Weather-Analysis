from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

def evaluate_models(models, X_test, y_test):
    results = []

    for name, model in models.items():
        predictions = model.predict(X_test)

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "MSE": mean_squared_error(y_test, predictions),
            "R2 Score": r2_score(y_test, predictions)
        })

    return pd.DataFrame(results)