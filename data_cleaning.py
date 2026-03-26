import pandas as pd
from tree import gini_index

data = pd.read_csv('mushrooms.csv', header= 0)

dane = pd.DataFrame(data)

print(dane.head(5))
