from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load(r'C:\Users\Parth\Desktop\DataScience\LinearRegression\stock_price_prediction\model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from request form
    open_val = float(request.form['open'])
    high_val = float(request.form['high'])
    low_val = float(request.form['low'])
    volume_val = float(request.form['volume'])

    # Prepare input data as a list
    input_data = [[open_val, high_val, low_val, volume_val]]

    # Make prediction
    prediction = model.predict(input_data)

    # Render prediction template with the result
    return render_template('predictions.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
