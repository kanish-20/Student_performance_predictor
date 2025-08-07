from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load('student_model.pkl')
scaler = joblib.load('scaler.pkl')
le_parent = joblib.load('label_encoder.pkl')
le_result = joblib.load('target_encoder.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            study_hours = float(request.form['study_hours'])
            attendance = float(request.form['attendance'])
            internal_marks = float(request.form['internal_marks'])
            parent_education = request.form['parent_education']

            # Encode & scale
            parent_encoded = le_parent.transform([parent_education])[0]
            features = np.array([[study_hours, attendance, internal_marks, parent_encoded]])
            scaled = scaler.transform(features)

            pred = model.predict(scaled)
            result = le_result.inverse_transform(pred)[0]

            return render_template('index.html', prediction=result)
        except Exception as e:
            return f"Error: {e}"
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
