from flask import Flask, request, render_template
import joblib
import pandas as pd
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup

# Load the model and vectorizer
model = joblib.load('phishing_email_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

app = Flask(__name__)

# Function to extract email content
def extract_email_content(eml_file):
    msg = BytesParser(policy=policy.default).parse(eml_file)
    email_body_html = msg.get_body(preferencelist=('html')).get_content()
    soup = BeautifulSoup(email_body_html, 'html.parser')
    return soup.get_text()

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Debugging: Print uploaded files to the console
    print(request.files)

    # Check if the file is in the request
    if 'file' not in request.files:
        return "No file part in the request"
    
    eml_file = request.files['file']

    # Check if a file was selected
    if eml_file.filename == '':
        return "No selected file"

    # Extract email content
    email_content = extract_email_content(eml_file)
    
    # Preprocess email content using the vectorizer
    X_tfidf = vectorizer.transform([email_content])
    
    # Combine TF-IDF features with the 'urls' feature (adding 'urls' placeholder as 0)
    X_combined = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
    X_combined['urls'] = 0  # Placeholder for the 'urls' feature
    
    # Make prediction
    prediction = model.predict(X_combined)
    result = 'Phishing' if prediction == 1 else 'Legitimate'
    
    # Render result on an HTML page
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
