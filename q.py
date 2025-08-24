# | xi      | yi      | xi * yi         | xi^2  |
# | ------- | ------- | --------------- | -------- |
# | 10      | 25      | 250             | 100      |
# | 15      | 30      | 450             | 225      |
# | 20      | 40      | 800             | 400      |
# | 25      | 45      | 1125            | 625      |
# | 30      | 50      | 1500            | 900      |
# | 35      | 60      | 2100            | 1225     |
# | 40      | 65      | 2600            | 1600     |
# | 45      | 70      | 3150            | 2025     |
# | 50      | 80      | 4000            | 2500     |
# | **Σ**   | **Σ**   | **Σ**           | **Σ**    |
# | **270** | **465** | **17975**       | **9600** |

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


data = {
    "x_i": [10, 15, 20, 25, 30, 35, 40, 45, 50],
    "y_i": [25, 30, 40, 45, 50, 60, 65, 70, 80]
}

df = pd.DataFrame(data)

df["x_i * y_i"] = df["x_i"] * df["y_i"]
df["x_i^2"] = df["x_i"] ** 2

totals = df.sum()

X = df[["x_i"]]
y = df["y_i"]
model = LinearRegression()
model.fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_

prediction_60 = model.predict([[60]])[0]
required_investment = (100 - intercept) / slope

print("Data Frame:")
print(df)
print("sum:")
print(totals)
print(f"model: y = {intercept:.2f} + {slope:.2f} * x")
print(f"60000: {prediction_60:.2f}")
print(f"100000: {required_investment:.2f}")