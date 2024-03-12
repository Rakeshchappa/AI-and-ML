import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Loan_Data.csv')
print(df.head(20).to_string())
print(df.shape)
print(df.isna().sum());


# Applying label encoding for few columns
def label_encode_columns(dataframe, columns):
    label_encoder = LabelEncoder()
    for column in columns:
        dataframe[column] = label_encoder.fit_transform(dataframe[column])
    return dataframe


# List of columns to apply label encoding
columns_to_encode = ['Education', 'Self_Employed', 'Married', 'Gender', 'Loan_Status']

# Applying label encoding to the specified columns
# df = label_encode_columns(df, columns_to_encode)

# Print the modified DataFrame
print(df.head(20).to_string())


# ****************************


def one_hot_encode_columns(dataframe, columns):
    encoded_df = pd.get_dummies(dataframe, columns=columns, prefix=columns)
    new_columns = encoded_df.columns
    return encoded_df, new_columns


# List of columns to apply one-hot encoding
columns_to_one_hot_encode = ['Education', 'Self_Employed', 'Married', 'Gender', 'Property_Area']

# Applying one-hot encoding to the specified columns
df, new_col = one_hot_encode_columns(df, columns_to_one_hot_encode)

# Print the modified DataFrame
print(df.head(20).to_string())
print(new_col)

# *******************
#  applying ONE hot Encoding
# Applying one-hot encoding to the "Property_Area" column
# df = pd.get_dummies(df, columns=['Property_Area'], prefix='Property_Area')
# df[['Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban']] = df[
#     ['Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban']].astype(int)
print(df.head(20).to_string())
# Reverse the label encoding for 'Education'
# Mapping 'Graduate' to 1 and 'Not Graduate' to 0
# df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
# print(df.head(20).to_string());
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
one_code_feature_to_pass_model = df[['Credit_History',
                                     'Education_Graduate', 'Education_Not Graduate', 'Self_Employed_No',
                                     'Self_Employed_Yes', 'Married_No', 'Married_Yes', 'Gender_Female',
                                     'Gender_Male', 'Property_Area_Rural', 'Property_Area_Semiurban',
                                     'Property_Area_Urban']]
#
# features = df[['Education', 'Self_Employed', 'ApplicantIncome', 'Credit_History', 'Property_Area_Rural',
# #                'Property_Area_Semiurban', 'Property_Area_Urban']]
# features = df[['Education', 'Self_Employed', 'ApplicantIncome', 'Credit_History', 'Property_Area']]
target = df['Loan_Status']
# features_encoded = pd.get_dummies(features, columns=['Education', 'Self_Employed', 'Property_Area'], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(one_code_feature_to_pass_model, target, test_size=0.2,
                                                    random_state=42)
print(f'X_train::{X_train}')
print(f'X_test::{X_test}')
print(f'y_train::{y_train}')
print(f'y_test::{y_test}')
# Random forest
clf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
# Train the classifier on the training data
clf_model.fit(X_train, y_train)
print('Model Trained sucessfully...')
# Make predictions on the test data
y_pred = clf_model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("RANDOM FOREST:Accuracy:", accuracy)
