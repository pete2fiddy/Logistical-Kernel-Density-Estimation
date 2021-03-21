from load_data import load_clean_airport
import numpy as np
import pandas as pd
df = load_clean_airport()
pd.set_option("display.max_rows", None, "display.max_columns", None)
df = df.drop(["year", "month", "destination", "day", "id", "schedule", "departure", "airline", "snowfall-amount"], axis = 1)

print(df.corr(method ='pearson'))
