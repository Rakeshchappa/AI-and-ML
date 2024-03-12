import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

df = pd.read_csv('Loan_Data.csv')
print(df.head(20).to_string())
print(df.shape)
print(df.isna().sum());
# visulzing the null values
missing_values = pd.DataFrame((df.isnull().sum() / len(df)) * 100, columns=['Percentage'])
missing_values.plot(kind='bar', title='Missing Values', ylabel='Percentage')
plt.show()


# DATA PRE PROCESSING ? FEATURE ENGEERNING

def Data_Imputation(df):
    parameters = {}
    for col in df.columns[df.isnull().any()]:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64' or df[col].dtype == 'int32':
            strategy = 'mean'
        else:
            strategy = 'most_frequent'
        missing_values1 = df[col][df[col].isnull()].values[0]
        parameters[col] = {'missing_values': missing_values1, 'strategy': strategy}
    print(parameters)
    return parameters


parameters_for_Data_cleaning = Data_Imputation(df);
for col, param in parameters_for_Data_cleaning.items():
    miss_val = param['missing_values']
    strategy = param['strategy']
    imp = SimpleImputer(missing_values=miss_val, strategy=strategy)
    df[[col]] = imp.fit_transform(df[[col]].values);
print("Model Features")
print(df.head(10).to_string());
print(df.shape)
features = df[['Education', 'Self_Employed', 'ApplicantIncome', 'Credit_History', 'Property_Area']]
target = df['Loan_Status']
features_encoded = pd.get_dummies(features, columns=['Education', 'Self_Employed', 'Property_Area'], drop_first=True)
print(df.head(20).to_string());
X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)
print(f'X_train::{X_train}')
print(f'X_test::{X_test}')
print(f'y_train::{y_train}')
print(f'y_test::{y_test}')
# Random forest
# Create a k-NN classifier with k=3
clf = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("KNN:Accuracy:", accuracy)
