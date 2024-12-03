from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the model and vectorizer
model = joblib.load('phishing_email_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to extract email content
def extract_email_content(eml_file_path):
    with open(eml_file_path, 'rb') as file:
        msg = BytesParser(policy=policy.default).parse(file)
    email_body_html = msg.get_body(preferencelist=('html')).get_content()
    soup = BeautifulSoup(email_body_html, 'html.parser')
    email_body_text = soup.get_text()
    return email_body_text

# Step 1: Extract content from multiple emails
email_1 = extract_email_content('Best way to use subtitles to learn Japanese.eml')
email_2 = extract_email_content('Thank you for reviewing Persona 3 Reload PS5 on Amazon.eml')

# Step 2: Label the emails (1 for phishing, 0 for legitimate)
# For the sake of this example, let's assume both emails are legitimate
emails = [
    {"body": email_1, "urls": 0, "label": 0},  # Assuming legitimate
    {"body": email_2, "urls": 0, "label": 0}   # Assuming legitimate
]

# Step 3: Create a DataFrame
df_emails = pd.DataFrame(emails)

# Step 4: Preprocess the email text using the TF-IDF vectorizer
X_tfidf = vectorizer.transform(df_emails['body'])

# Step 5: Combine the TF-IDF features with the 'urls' feature
X_combined = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
X_combined['urls'] = df_emails['urls'].values

# Step 6: Predict using the trained model
y_pred = model.predict(X_combined)

# Step 7: Add predictions to the DataFrame
df_emails['prediction'] = y_pred
df_emails['prediction'] = df_emails['prediction'].apply(lambda x: 'Phishing' if x == 1 else 'Legitimate')

# Step 8: Evaluate the model (Optional, since we're using a small dataset)
accuracy = accuracy_score(df_emails['label'], y_pred)
conf_matrix = confusion_matrix(df_emails['label'], y_pred)
class_report = classification_report(df_emails['label'], y_pred, zero_division=1)


# Display the results
print(df_emails)
print(f'\nAccuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
