import streamlit as st
import numpy as np
import pandas as pd

st.header("My first ADVERTISING App")
st.write(pd.DataFrame({
    'Intplan': ['yes', 'yes', 'yes', 'no'],
    'Churn Status': [0, 0, 0, 1]
}))
