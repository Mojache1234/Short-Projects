### Using Libraries

# import pandas as pd
# from sklearn.linear_model import LinearRegression

# df = pd.read_csv('laptop_battery.csv')

# X = df.battery_life.values.reshape(-1, 1)
# y = df.laptop_hours.values
# lm = LinearRegression()

# lm.fit(X, y)
# y_pred = lm.predict([[1.5]])
# print(y_pred)

### Reading from CSV w/o sklearn

# import csv
# import math

# X = []
# y = []

# with open('laptop_battery.csv') as f:
#     f.readline() # skip column headers
#     reader = csv.reader(f)
#     for row in reader:
#         X.append(float(row[0]))
#         y.append(float(row[1]))

# # Train-Test split
# X_train = X[:76]
# X_test = X[76:100]
# y_train = y[:76]
# y_test = y[76:]

# # Variables
# s1 = sum(X_train)
# s2 = sum([x**2 for x in X_train])
# Cxy = sum([a*b for a, b in zip(X_train, y_train)])
# Cy = sum(y_train)
# n = len(X_train)

# # Linear Regression Variables
# m = (n*Cxy - s1*Cy) / (n*s2 - s1**2)
# b = (s2*Cy - s1*Cxy) / (n*s2 - s1**2)

# # Prediction
# y_pred = [m * x + b for x in X_test]
# for a, b in zip(y, y_pred):
#     print(a, ', ', b)

# # Root Mean-Square Error
# RMSE = math.sqrt(sum([(a - b)**2 for a, b in zip(y, y_pred)]))
# print(RMSE)

### Exploring the data
# import pandas as pd
# from matplotlib import pyplot as plt

# df = pd.read_csv('laptop_battery.csv')

# plt.scatter(df.battery_life, df.laptop_hours)
# plt.show()

### Reading from CSV w/o sklearn - Using Polynomial

# import csv
# import math

# X = []
# y = []

# with open('laptop_battery.csv') as f:
#     f.readline() # skip column headers
#     reader = csv.reader(f)
#     for row in reader:
#         X.append(float(row[0]))
#         y.append(float(row[1]))

# # Create square input
# X = [(x, x**2) for x in X]

# # Train-Test split
# X_train = X[:76]
# X_test = X[76:100]
# y_train = y[:76]
# y_test = y[76:]

# # Variables
# s1_0 = sum([x[0] for x in X_train])
# s2_0 = sum([x[0]**2 for x in X_train])
# Cxy_0 = sum([a[0]*b for a, b in zip(X_train, y_train)])

# s1_1 = sum([x[1] for x in X_train])
# s2_1 = sum([x[1]**2 for x in X_train])
# Cxy_1 = sum([a[1]*b for a, b in zip(X_train, y_train)])

# Cy = sum(y_train)
# n = len(X_train)

# # Linear Regression Model
# m0 = (n*Cxy_0 - s1_0*Cy) / (n*s2_0 - s1_0**2)
# m1 = (n*Cxy_1 - s1_1*Cy) / (n*s2_1 - s1_1**2)
# b = (s2_0*Cy - s1_0*Cxy_0) / (n*s2_0 - s1_0**2)

# # Prediction
# y_pred = [m0 * x + m1 * x**2 + b for x in X_test]
# for a, b in zip(y, y_pred):
#     print(a, ', ', b)

# # Root Mean-Square Error
# RMSE = math.sqrt(sum([(a - b)**2 for a, b in zip(y, y_pred)]))
# print(RMSE)

# Polynomial w/ Scikit Learn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import math

df = pd.read_csv('laptop_battery.csv')

X_train, X_test, y_train, y_test = train_test_split(df.battery_life.values, df.laptop_hours.values, test_size=.25)

# Simple Linear Model
lm1 = LinearRegression()
lm1.fit(X_train.reshape(-1, 1), y_train)
y_pred1 = lm1.predict(X_test.reshape(-1, 1))
RSME1 = math.sqrt(mean_squared_error(y_test, y_pred1))

# Polynomial Linear Model
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
X_test_poly = poly.fit_transform(X_test.reshape(-1, 1))

lm2 = LinearRegression()
lm2.fit(X_train_poly, y_train)
y_pred2 = lm2.predict(X_test_poly)
y_train_pred = lm2.predict(X_train_poly)
RSME2 = math.sqrt(mean_squared_error(y_test, y_pred2))

# Plot
plt.scatter(df.battery_life.values, df.laptop_hours.values, color='blue')

plt.plot(X_test, y_pred1, color='red')

predicted_data = sorted(list(zip(X_train, y_train_pred)) + list(zip(X_test, y_pred2)))

plt.plot(
    [x[0] for x in predicted_data],
    [x[1] for x in predicted_data],
    color='green'
)

plt.show()

# print(lm1.coef_)
# print(lm1.intercept_)

# print(lm2.coef_)
# print(lm2.intercept_)