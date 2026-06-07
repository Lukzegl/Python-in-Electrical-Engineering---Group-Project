import matplotlib.pyplot as plt
import pandas as pd

df1 = pd.read_csv('data/raw/ETTm2.csv')


df1.drop(columns=['date'], inplace=True) 

korelacja = df1.corr()
print(korelacja)



