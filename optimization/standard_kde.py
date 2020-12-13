from load_data import load_clean_airport
import scipy.stats as stats
import numpy as np
def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]
df = load_clean_airport()
y = df["delayed"]
X = df[df["airline"]=="Iberia"]
X = X.drop(["delayed", "year", "month", "destination", "id", "schedule", "departure", "airline"], axis = 1)
X = X.to_numpy().T
print(X)
kde = stats.gaussian_kde(X)
