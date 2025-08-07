import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv('student_performance.csv')
print("📦 Columns:", df.columns)

# Clean and check
df.dropna(inplace=True)
print("✅ Missing values:\n", df.isnull().sum())
print("✅ Shape after cleanup:", df.shape)

# Encode target and categorical
le_parent = LabelEncoder()
le_result = LabelEncoder()

df['parent_education'] = le_parent.fit_transform(df['parent_education'])
df['result'] = le_result.fit_transform(df['result'])

# Features & target
X = df[['study_hours', 'attendance', 'internal_marks', 'parent_education']]
y = df['result']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split & train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'student_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_parent, 'label_encoder.pkl')
joblib.dump(le_result, 'target_encoder.pkl')

print("✅ Model and files saved.")
