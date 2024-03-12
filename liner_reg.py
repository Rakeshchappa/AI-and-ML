# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
#
# # Read the CSV file into a DataFrame
# df = pd.read_csv('salary.csv')
# #
# # # Create a scatter plot using DataFrame's plotting functionality
# # df.plot.scatter(x='Experience', y='Salary')
# #
# # # Set labels and title
# # plt.xlabel('Experience')
# # plt.ylabel('Salary')
# # plt.title('Scatter Plot of Experience vs Salary')
#
# # Show the plot
# # # plt.show()
# # X = df['Experience']
# X = df[['Experience']]
# y = df['Salary']
# X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.1)
# print(X_train);
# model = LinearRegression();
# model.fit(X_train, Y_train);
# print("model trained sucessfully")
# Y_predcit = model.predict(X_test);
# print(f"predicted values{Y_predcit}")
# print(f"Y_text values{Y_test}")
# plt.scatter(Y_predcit, Y_test)
# plt.show();
# print(model.score(X, y))
# # ***************************************

from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the provided dataset
df = pd.read_csv('TSLA.csv')

# Select features and target variable
features = df[['Open', 'High', 'Low', 'Volume']]
target = df['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'R-squared Score: {r2}')

# Visualize the predicted vs actual values
plt.scatter(y_test, predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()
print(y_test)
count_for_y_test=0
# for i in y_test:
#     print(i)
#     count_for_y_test+=1
# count=0
# print("HELLOO RAKESHHH**********************************************************************************")
# for j in predictions:
#     print(j)
#     count+=1
# print(f"y-testvalues{count_for_y_test}::: predicted values::{count}")
count_for_y_test = 0

# Printing side by side using zip
for actual, predicted in zip(y_test, predictions):
    # print(f"Actual: {actual}, Predicted: {predicted}")
    print(f"Actual: {round(actual, 6)}, Predicted: {round(predicted, 6)}")
    count_for_y_test += 1

print(f"Total values: {count_for_y_test}")


# Assume 'new_record' is a dictionary or DataFrame with the same columns as your original dataset
new_record = {
    'Open': 22.0,
    'High': 23.5,
    'Low': 21.8,
    'Volume': 12000000
}

# Convert the new record to a DataFrame
new_df = pd.DataFrame([new_record])

# Use the trained model to make predictions
predicted_close = model.predict(new_df)

print(f'Predicted Close Price for the New Record: {predicted_close[0]}')
