import pandas as pd

def load_and_clean_data():
    df = pd.read_csv("data/weather.csv")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Rename columns to standard format
    df.rename(columns={
        "Date": "date",
        "Temperature_C": "temperature"
    }, inplace=True)

    # Convert date
    df['date'] = pd.to_datetime(df['date'])

    # Drop missing values
    df = df.dropna()

    # Feature engineering
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    return df