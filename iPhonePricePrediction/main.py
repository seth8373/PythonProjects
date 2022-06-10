import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

iData = pd.read_csv("iphone_price.csv")
# Visualize the data
plt.scatter(iData['version'], iData['price'])
plt.xlabel('version');
plt.ylabel("price ($)")
plt.show()

# Create Machine Learning Model
# Regression model is of form y=mx+b; where m is the coefficient/gradient and b is the intercept
model = LinearRegression()

model.fit(iData[['version']], iData[['price']])

# Determine model coefficient x and intercept and coefficient of determination
print(model.coef_)
print(model.intercept_)
print(model.score(iData[['version']], iData[['price']]))

# Make Prediction

print(model.predict([[14]]))
