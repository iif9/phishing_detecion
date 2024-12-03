import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib  # Import joblib to save the model and vectorizer

# Load multiple datasets
df1 = pd.read_csv('/home/kali/Desktop/datasets/CEAS_08.csv')
df2 = pd.read_csv('/home/kali/Desktop/datasets/phishing_email.csv')  # Add the path to your new dataset
df3 = pd.read_csv('/home/kali/Desktop/datasets/Nigerian_Fraud.csv')  # Add another dataset if you have more

# Combine the datasets
df = pd.concat([df1, df2, df3], ignore_index=True)

# Optionally use a smaller subset of the data for testing (e.g., 10,000 rows)
df = df.sample(n=10000, random_state=42)

# Inspect the first few rows of the dataset
print(df.head())

# Get information about the dataset
print(df.info())

# Check for missing values in the dataset
print(df.isnull().sum())

# Convert the 'date' column to a datetime format
df['date'] = pd.to_datetime(df['date'], format='%a, %d %b %Y %H:%M:%S %z', errors='coerce', utc=True)

# Check the datatype to confirm conversion
print(df['date'].dtype)

# Create separate columns for 'Date' and 'Time'
df['Date'] = df['date'].dt.date
df['Time'] = df['date'].dt.time

# Drop rows with missing or invalid dates
df = df.dropna(subset=['date'])

# Display the first few rows to check the changes
print(df[['date', 'Date', 'Time']].head())

# Fill missing values in the 'receiver' column with 'Unknown'
df['receiver'].fillna('Unknown', inplace=True)

# Fill missing values in the 'subject' column with an empty string
df['subject'].fillna('', inplace=True)

# Verify that there are no more missing values
print(df.isnull().sum())

# Preprocess the email body text using TF-IDF vectorization with fewer features
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Transform the email body text into TF-IDF features
X_tfidf = tfidf_vectorizer.fit_transform(df['body'])

# Convert the TF-IDF matrix to a DataFrame and add other features
features = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Add the 'urls' column to the features DataFrame
features['urls'] = df['urls'].values

# Add the 'label' column as the target variable
features['label'] = df['label'].values

# Ensure there are no NaN values in the features DataFrame
features.dropna(inplace=True)

# Split the data into features and target variable
X = features.drop('label', axis=1)
y = features['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the trained model and vectorizer
joblib.dump(model, 'phishing_email_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("Model and vectorizer have been saved successfully.")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Add predictions back to the original DataFrame
df_test = df.iloc[X_test.index].copy()
df_test['predicted_label'] = y_pred

# Display emails with their actual and predicted labels
result_df = df_test[['body', 'label', 'predicted_label']]
result_df['label'] = result_df['label'].apply(lambda x: 'Phishing' if x == 1 else 'Legitimate')
result_df['predicted_label'] = result_df['predicted_label'].apply(lambda x: 'Phishing' if x == 1 else 'Legitimate')

print(result_df.head(10))  # Display the first 10 rows as an example

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, zero_division=1)

# Print the evaluation results
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Now, show all the plots at the end
# Visualize the distribution of phishing (1) vs. legitimate (0) emails
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='label', data=df)
plt.title('Distribution of Phishing (1) vs. Legitimate (0) Emails')
plt.xlabel('Label')
plt.ylabel('Count')

# Add percentages on top of bars
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height/len(df):.2%}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 8),
                textcoords='offset points')

plt.show()

# Annotate the Confusion Matrix with explanations next to the numbers
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues', cbar=False)

# Add explanations next to the numbers
labels = np.array([['True Negatives (TN)', 'False Positives (FP)'],
                   ['False Negatives (FN)', 'True Positives (TP)']])

for i in range(2):
    for j in range(2):
        plt.text(j + 0.5, i + 0.5, f'{conf_matrix[i, j]}\n{labels[i, j]}',
                 ha='center', va='center', fontsize=12, color='black', weight='bold')

plt.title('Confusion Matrix with Explanations')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
