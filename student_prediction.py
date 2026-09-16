import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("students_100.csv")

print("Total students:", len(data))

X = data[
    [
        "hours_studied",
        "attendance",
        "previous_score"
    ]
]

y = data["passed"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", len(X_train))
print("Testing data:", len(X_test))

model = LogisticRegression()

model.fit(X_train, y_train)

print("Model trained successfully!")

test_predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    test_predictions
)

print("Test Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions
    )
)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross-validation scores:")
print(cv_scores)

print("Average CV Accuracy:", cv_scores.mean())

joblib.dump(
    model,
    "student_model.pkl"
)

print("\nModel saved successfully!")