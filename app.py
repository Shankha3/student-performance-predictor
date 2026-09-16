import streamlit as st
import joblib
import pandas as pd

model = joblib.load("student_model.pkl")

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Predictor")
st.write("Enter the student's academic information to get a prediction.")

st.divider()

st.subheader("📊 Student Information")

hours_studied = st.number_input(
    "Hours Studied per Day",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)

previous_score = st.number_input(
    "Previous Score (%)",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=1.0
)

st.divider()

if st.button("🔮 Predict Performance", use_container_width=True):

    if attendance < 50:
        st.warning("⚠️ Attendance is below 50%.")

    if previous_score < 35:
        st.warning("⚠️ Previous score is below 35%.")

    new_student = pd.DataFrame(
        [[hours_studied, attendance, previous_score]],
        columns=[
            "hours_studied",
            "attendance",
            "previous_score"
        ]
    )

    prediction = model.predict(new_student)[0]
    probability = model.predict_proba(new_student)[0]

    st.subheader("📋 Prediction Summary")

    st.write(f"Hours Studied: **{hours_studied} hours/day**")
    st.write(f"Attendance: **{attendance:.0f}%**")
    st.write(f"Previous Score: **{previous_score:.0f}%**")

    if prediction == 1:
        pass_probability = probability[1] * 100

        st.success("### ✅ Student is likely to PASS")

        st.write(
            f"**Model confidence for PASS: {pass_probability:.2f}%**"
        )

        st.progress(
            int(pass_probability),
            text=f"PASS Probability: {pass_probability:.2f}%"
        )

    else:
        fail_probability = probability[0] * 100

        st.error("### ❌ Student is likely to FAIL")

        st.write(
            f"**Model confidence for FAIL: {fail_probability:.2f}%**"
        )

        st.progress(
            int(fail_probability),
            text=f"FAIL Probability: {fail_probability:.2f}%"
        )

st.divider()

st.caption(
    "Note: This application uses a synthetic educational dataset "
    "and is intended for demonstration purposes."
)