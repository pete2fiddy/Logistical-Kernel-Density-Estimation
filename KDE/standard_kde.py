from KDE.load_data import load_kde_cleaned_airline_data
import scipy.stats as stats
import numpy as np
'''
Returns scipy standard gaussian KDE for the airline with name airline_name
'''
def get_standard_kde(airline_name):
    X = load_kde_cleaned_airline_data(airline_name)
    X = X.to_numpy().T
    kde = stats.gaussian_kde(X)
    return kde

if __name__ == "__main__":
    kde = get_standard_kde("Iberia")
