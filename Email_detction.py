import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create a DataFrame
# Read from JSON file
df = pd.read_json('email_data.json')

# Display the DataFrame
print(df)
# Check lengths
print(len(df['email_text']), len(df['label']))
# Step 3: Feature Extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['email_text'])
print(f"Rakesh chappa:{X}")
# Step 4: Splitting the Datas
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

# Step 5: Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 8: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Additional metrics
print(classification_report(y_test, y_pred))


# Assuming 'new_email_text' is the text of the email you want to classify
new_email_text = ["your statement is available"]

# Transform the new email text using the same CountVectorizer used during training
new_email_vectorized = vectorizer.transform(new_email_text)

# Make a prediction using the trained logistic regression model
prediction = model.predict(new_email_vectorized)

# Interpret the prediction
if prediction[0] == 1:
    print("This email is classified as spam.")
else:
    print("This email is not classified as spam.")
