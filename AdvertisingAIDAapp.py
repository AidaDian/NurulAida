import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Advertising Prediction App

This app predicts the **ADVERTISING AIDA** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 1.0, 200.0, 100.0)
    Radio = st.sidebar.slider('Radio', 1.0, 40.0, 20.0)
    Newspaper = st.sidebar.slider('Newspaper', 1.0, 100.0, 50.0)
    Sales = st.sidebar.slider('Sales', 1.0, 10.0, 20.0)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper,
            'Sales': Sales}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

Advertising = datasets.Advertising()
X = Advertising.data
Y = Advertising.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(Advertising.target_names)

st.subheader('Prediction')
st.write(Advertising.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
