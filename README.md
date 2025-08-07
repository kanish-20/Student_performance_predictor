# 🎓 Student Performance Predictor

This is a Flask-based web application that uses a **Random Forest Classifier** to predict whether a student will **Pass or Fail** based on their academic performance and background.

---

## 🚀 Features

- Predicts student performance (Pass/Fail)
- Machine Learning model using Random Forest
- Scaled and encoded data handling
- Web interface for input
- Custom styling with HTML + CSS

---

## 📂 Dataset

The model is trained on a custom dataset with the following features:

- `study_hours` — Number of hours a student studies per day
- `attendance` — Attendance percentage
- `internal_marks` — Internal exam marks
- `parent_education` — Parent’s education level (`Graduate`, `Not Graduate`)
- `result` — Target label (`Pass`, `Fail`)

---

## 🧠 Model

- Algorithm: `RandomForestClassifier`
- Libraries: `scikit-learn`, `pandas`, `numpy`
- Preprocessing:
  - Missing value handling
  - Label Encoding (`parent_education`, `result`)
  - Feature Scaling using `StandardScaler`

---

## 🖥️ Web App

### Tech Stack:

- `Flask`
- `HTML/CSS`
- `Python`
- `Joblib` (for saving and loading models)

### Usage:

1. User enters input data via form
2. App preprocesses and encodes input
3. Model predicts result
4. Result displayed as **"Pass"** or **"Fail"**

---

## 📦 Installation

```
git clone https://github.com/your-username/student-performance-predictor.git
cd student-performance-predictor
pip install -r requirements.txt
```

## Run the app

```
python train_model.py   # Train the model and save it
python app.py           # Start the Flask server
```

## Open browser

```
http://127.0.0.1:5000/
```

## Project structure

```
student-performance-predictor/
├── app.py
├── train_model.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── student_data.csv
├── scaler.pkl
├── model.pkl
├── label_encoder.pkl
├── target_encoder.pkl
└── requirements.txt
```

