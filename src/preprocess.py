import pandas as pd

def load_and_clean_data():
    df = pd.read_csv("data/weather.csv")

    
    df.columns = df.columns.str.strip()

   
    df.rename(columns={
        "Date": "date",
        "Temperature_C": "temperature"
    }, inplace=True)

   
    df['date'] = pd.to_datetime(df['date'])

    
    df = df.dropna()

    # Feature engineering
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    return df
