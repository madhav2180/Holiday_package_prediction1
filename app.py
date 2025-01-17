from flask import Flask, request, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the encoder, scaler, and model
scaler_file = os.path.join('static', 'scaler.pkl')
encoder_file = os.path.join('static', 'encoder.pkl')
model_file = os.path.join('static', 'best_rf_model.pkl')

with open(scaler_file, 'rb') as f:
    scaler = pickle.load(f)

with open(encoder_file, 'rb') as f:
    encoder = pickle.load(f)

with open(model_file, 'rb') as f:
    model = pickle.load(f)

# Define the complete list of features used during training
required_features = ['Age', 'DurationOfPitch', 'PreferredPropertyStar', 'MonthlyIncome', 'TotalVisiting']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input features
        inputs = [float(x) for x in request.form.values()]
        input_df = pd.DataFrame([inputs], columns=required_features[:len(inputs)])

        # Handle missing features by adding them with default values
        for feature in required_features:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Default value, adjust as per your dataset

        # Ensure column order matches training
        input_df = input_df[required_features]

        # Scale the data
        scaled_data = scaler.transform(input_df)

        # Predict using the model
        prediction = model.predict(scaled_data)
        result = 'User is likely to take the Wellness Tourism Package' if prediction[0] == 1 else 'User is not likely to take the Wellness Tourism Package'
        result_class = 'positive-result' if prediction[0] == 1 else 'negative-result'

        return render_template('result.html', prediction_text= result, result_class=result_class)
    except Exception as e:
        return render_template('error.html', error_message=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
