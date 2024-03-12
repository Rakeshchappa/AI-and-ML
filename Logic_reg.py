import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv('insurance_data.csv')
print(df.head(20))
print(df.shape)
print(df.isna().sum());
# here features are age because based on the age we need set our traget
# variable and that we need to predict the traget value.

features = df[['age']]
traget = df['bought_insurance']
plt.scatter(features, traget)
plt.xlabel('Ages')
plt.ylabel('Bought Or noT')
plt.title('Insurance bought or not')
plt.show()
# split the data to train the model
# x_train, y_train, x_test, y_test = train_test_split(features, traget, test_size=0.2, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(features, traget, test_size=0.2, random_state=42)
model = LogisticRegression()
# print(f"x_train:::{x_train}")
# print(f"y_train:::{y_train}")
# If you print x_test without using .values, it will display the DataFrame with the index and column names
# print(f"x_test:::{x_test.values}")
# print(f"y_test:::{y_test.values}")
model.fit(x_train, y_train);
Insurance_prediction_model = model.predict(x_test)
print(f"rakesh.............{Insurance_prediction_model}")
print(f"{x_test.values}:::::Bought or no{Insurance_prediction_model}")
accuracy = accuracy_score(y_test, Insurance_prediction_model)
print(f"Accuracy: {accuracy}")
plt.scatter(x_test,y_test)
plt.show();

def checking_insurance_using_model(age):
    age_reshaped = [[age]]

    prediction = model.predict(age_reshaped)

    # Display the result
    if prediction == 1:
        return "Insurance will be bought"
    else:
        return "Insurance will not be bought"


# Keep asking for input until the user enters "exit"
while True:
    user_input = input("Please enter the age to predict if insurance will be bought or enter 'exit' to exit: ")

    if user_input.lower() == 'exit':
        print("Thank you")
        break

    try:
        user_age = float(user_input)
        result = checking_insurance_using_model(user_age)
        print(result)
    except ValueError:
        print("Invalid input. Please enter a valid age or 'exit'.")
