from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load and train the model
data = pd.read_csv('emails.csv')
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']

model = MultinomialNB()
model.fit(X, y)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email_text = request.form['email']
        email_vec = vectorizer.transform([email_text])
        prediction = model.predict(email_vec)
        result = 'Spam' if prediction[0] == 'spam' else 'Ham'
        return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
