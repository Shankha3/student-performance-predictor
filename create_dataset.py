import pandas as pd
import random

random.seed(42)

students = []

for i in range(100):
    hours_studied = random.randint(1, 10)
    attendance = random.randint(50, 100)
    previous_score = random.randint(35, 95)

    # Calculate a performance score
    score = (
        hours_studied * 4
        + attendance * 0.3
        + previous_score * 0.5
    )

    # Decide pass/fail
    if score >= 70:
        passed = 1
    else:
        passed = 0

    students.append([
        hours_studied,
        attendance,
        previous_score,
        passed
    ])

# Create DataFrame
data = pd.DataFrame(
    students,
    columns=[
        "hours_studied",
        "attendance",
        "previous_score",
        "passed"
    ]
)

# Save dataset
data.to_csv("students_100.csv", index=False)

print("Dataset created successfully!")
print("Total students:", len(data))
print(data.head(10))