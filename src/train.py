from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

def train_models(df):
    X = df[['day', 'month', 'year', 'Humidity_%', 'Wind_Speed_km_h', 'Precipitation_mm']]
    y = df['temperature']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor()
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    # Save best model
    with open("models/model.pkl", "wb") as f:
        pickle.dump(trained_models["RandomForest"], f)

    return trained_models, X_test, y_test