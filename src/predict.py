import pickle

def load_model():
    with open("models/model.pkl", "rb") as f:
        return pickle.load(f)

def predict_temp(day, month, year, humidity, wind, precipitation):
    model = load_model()
    return model.predict([[day, month, year, humidity, wind, precipitation]])[0]