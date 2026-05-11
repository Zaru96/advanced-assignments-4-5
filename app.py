import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('model.joblib')

st.title("Titanic Survival Prediction")
st.write("Prediksi keselamatan penumpang Titanic")

# Input user
pclass = st.selectbox("Passenger Class", [1, 2, 3])

sex = st.selectbox("Sex", ["male", "female"])

sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)

parch = st.number_input("Parents/Children Aboard", 0, 10, 0)

fare = st.number_input("Fare", 0.0, 600.0, 50.0)

embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# Dataframe input
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked]
})

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")
