
import streamlit as st
import pickle
import numpy as np

model = pickle.load(
    open("logistic_model.pkl", "rb")
)

st.title("Titanic Survival Prediction")

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    [0, 1]
)

age = st.number_input(
    "Age",
    min_value=0
)

sibsp = st.number_input(
    "SibSp",
    min_value=0
)

parch = st.number_input(
    "Parch",
    min_value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0
)

embarked = st.selectbox(
    "Embarked",
    [0, 1, 2]
)

input_data = np.array(
    [[
        pclass,
        sex,
        age,
        sibsp,
        parch,
        fare,
        embarked
    ]]
)

prediction = model.predict(input_data)

if st.button("Predict"):

    if prediction[0] == 1:
        st.success("Passenger Survived")

    else:
        st.error("Passenger Did Not Survive")
