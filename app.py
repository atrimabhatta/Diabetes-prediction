from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the dataset
url = 'diabetes - diabetes.csv'
df = pd.read_csv(url)

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('Outcome', axis=1))

# Create a new DataFrame with the scaled features
df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_scaled['Outcome'] = df['Outcome']

# Split the data into training and testing sets
X = df_scaled.drop('Outcome', axis=1)
y = df_scaled['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    final_features_scaled = scaler.transform(final_features)
    prediction = model.predict(final_features_scaled)
    output = 'This Is Diabetic' if prediction[0] == 1 else 'This Is Not Diabetic'
    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)