import pandas as pd

df = pd.read_csv("tuning_tests.csv")


print(len(df))  
#Remove duplicates rows
df = df.drop_duplicates()

df.to_csv("tuning_tests.csv", index=False)