from load_data import load_clean_airport
import scipy.stats as stats
import numpy as np
'''
Returns scipy standard gaussian KDE for the airline with name airline_name
'''
def get_standard_kde(airline_name):
    df = load_clean_airport()
    y = df["delayed"]
    X = df[df["airline"]==airline_name]
    #remove delayed for the y variable
    X = X.drop(["delayed", "year", "month", "destination", "id", "schedule", "departure", "airline", "snowfall-amount"], axis = 1)
    X = X.to_numpy().T
    kde = stats.gaussian_kde(X)
    return kde

if __name__ == "__main__":
    kde = get_standard_kde("Iberia")
