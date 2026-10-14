from flask import Flask, render_template, request
import numpy as np
import joblib

# Initialize Flask application
app = Flask(__name__)

# Load the trained machine learning model
obj = joblib.load('california.joblib')

model = obj['Model']
columns = obj['Columns']


# Home page
@app.route('/')
def Main():
    return render_template('index.html', columns=columns)


# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    try:
        Input = []

        # Collect input values from the HTML form
        for column in columns:
            value = request.form.get(column)

            if value is None or value.strip() == '':
                return render_template(
                    'index.html',
                    columns=columns,
                    error=f'Please enter a value for {column}.',
                    values=request.form
                )

            Input.append(float(value))

        # Predict house value
        out = model.predict(np.array([Input]))[0]

        # Display prediction on the same page
        return render_template(
            'index.html',
            columns=columns,
            prediction=round(float(out), 2),
            values=request.form
        )

    except (ValueError, TypeError):
        return render_template(
            'index.html',
            columns=columns,
            error='Please enter valid numerical values.',
            values=request.form
        )

    except Exception:
        app.logger.exception("Prediction failed")

        return render_template(
            'index.html',
            columns=columns,
            error='Unable to generate prediction. Please try again.',
            values=request.form
        )


# Run the application
if __name__ == '__main__':
    app.run(debug=True, port=5001)