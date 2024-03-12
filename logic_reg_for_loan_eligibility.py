import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

df = pd.read_csv('Loan_Data.csv')
print(df.head(20).to_string())
print(df.shape)
print(df.isna().sum());
# here features are age because based on the age we need set our traget
# variable and that we need to predict the traget value.
#
# features = df[['age']]
# traget = df['bought_insurance']
# plt.scatter(features, traget)
# plt.xlabel('Ages')
# plt.ylabel('Bought Or noT')
# plt.title('Insurance bought or not')
# plt.show()
# # split the data to train the model
# # x_train, y_train, x_test, y_test = train_test_split(features, traget, test_size=0.2, random_state=42)
# x_train, x_test, y_train, y_test = train_test_split(features, traget, test_size=0.2, random_state=42)
# model = LogisticRegression()

missing_values = pd.DataFrame((df.isnull().sum() / len(df)) * 100, columns=['Percentage'])
missing_values.plot(kind='bar', title='Missing Values', ylabel='Percentage')
plt.show()

# Data pre-processing/Data Cleaning using get dummies or simple imputer
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy='mean')
print(df.LoanAmount.isnull().sum())
df['LoanAmount'] = imp.fit_transform(df[['LoanAmount']])
print(df.LoanAmount.isnull().sum())


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
print(df.shape)
print(df.isna().sum());
# Assuming 'df' is your DataFrame containing the dataset
columns_to_encode = ['Self_Employed', 'Education', 'Married', 'Gender']  # Add other columns as needed
print("hiiii")
print(df['Credit_History'].unique());
print(df['Education'].unique());
print(df['Self_Employed'].unique())
print(df['Married'].unique());
print(df['Gender'].unique());
print(df['Property_Area'].unique());
# One-hot encode specified columns
df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)
features = df[['Education'], ['Self_Employed'], ['ApplicantIncome'], ['Credit_History'], ['Property_Area']]
target = df['Loan_Status']
# features = df[['LoanAmount']]
# **********************
# Scatter plot for visualization
plt.scatter(features, target)
plt.xlabel('Applicant Income')
plt.ylabel('Loan Status')
plt.title('Loan Eligibility Visualization')
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Build a Random Forest Classifier model
model = LogisticRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Logestic regression::Accuracy: {accuracy}")
print("Applying Dummies")
print(df.head(20).to_string())
print("end....")

#Random forest alog
clf_model=RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
# Train the classifier on the training data
clf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf_model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("RANDOM FOREST:Accuracy:", accuracy)
# k NN
# Create a k-NN classifier with k=3
clf = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("KNN:Accuracy:", accuracy)
