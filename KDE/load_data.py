import pandas as pd

def load_clean_airport():
    df=pd.read_csv('adolfosuarez20191008-20191027.csv', sep=',',header=0)
    df = df.replace('-', '', regex=True)
    df = df[df["sunshine-duration"] != -999]
    return df
