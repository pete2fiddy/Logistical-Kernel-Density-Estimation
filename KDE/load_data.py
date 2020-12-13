import pandas as pd

def load_clean_airport():
    df=pd.read_csv('KDE/adolfosuarez20191008-20191027.csv', sep=',',header=0)
    df = df.replace('-', '', regex=True)
    df = df[df["sunshine-duration"] != -999]
    return df


def load_kde_cleaned_airline_data(airline_name):
    df = load_clean_airport()
    y = df["delayed"]
    X = df[df["airline"]==airline_name]
    #remove delayed for the y variable
    X = X.drop(["delayed", "year", "month", "destination", "id", "schedule", "departure", "airline", "snowfall-amount"], axis = 1)
    return X
