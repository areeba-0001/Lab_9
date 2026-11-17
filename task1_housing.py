import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

homes = pd.read_csv("real_estate.csv")
homes["distance_100m"] = homes["X3 distance to the nearest MRT station"] / 100

X = homes[["distance_100m"]]
y = homes["Y house price of unit area"]

model = LinearRegression()
model.fit(X, y)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

new_house = pd.DataFrame({"distance_100m": [5]})
print("Predicted Price at 500m:", model.predict(new_house)[0])

plt.figure(figsize=(8, 5))
plt.scatter(X["distance_100m"], y, label="Observations")
plt.plot(X["distance_100m"], model.predict(X), color="red", label="Fitted Line")
plt.xlabel("Distance to MRT Station (100m)")
plt.ylabel("House Price per Unit Area")
plt.title("Housing Price Prediction")
plt.legend()
plt.grid()
plt.savefig("task1_housing.png", dpi=300, bbox_inches="tight")
plt.show()
