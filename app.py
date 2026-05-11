import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('model.joblib')

# Page Config
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# Title
st.title("🚢 Titanic Survival Prediction")
st.markdown("### Predict whether a passenger would survive the Titanic disaster")

st.write("---")

# Input Section
st.subheader("Passenger Information")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    sex = st.selectbox(
        "Sex",
        ["male", "female"]
    )

    embarked = st.selectbox(
        "Embarked",
        ["S", "C", "Q"]
    )

with col2:
    sibsp = st.number_input(
        "Siblings/Spouses Aboard",
        min_value=0,
        max_value=10,
        value=0
    )

    parch = st.number_input(
        "Parents/Children Aboard",
        min_value=0,
        max_value=10,
        value=0
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=50.0
    )

st.write("---")

# Create dataframe
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked]
})

# Predict Button
if st.button("Predict Survival"):

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Passenger Survived")
        st.balloons()
    else:
        st.error("❌ Passenger Did Not Survive")

# Footer
st.write("---")
st.caption("Machine Learning Inference App using Streamlit")
