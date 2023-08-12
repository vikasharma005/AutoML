import streamlit as st
import plotly.express as px 

import os
import pandas as pd

import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

from pycaret.regression import setup, compare_models, pull, save_model

with st.sidebar: 
    st.image("https://www.google.com/url?sa=i&url=https%3A%2F%2Fh2o.ai%2Fplatform%2Fh2o-automl%2F&psig=AOvVaw2pi1JIxgkeyZJECAOoC2vs&ust=1691897765696000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCMinl7-Y1oADFQAAAAAdAAAAABAD")
    st.title("Auto Streamlit")
    choice = st.radio("Navigation", ["Upload", "Profile", "Machine Learning", "Download"])
    st.info("Hello")

if os.path.exists("dataset.csv"): 
    df = pd.read_csv('dataset.csv', index_col=None)


if choice == "Upload": 
    st.title("Upload")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv("dataset.csv", index=None)
        st.dataframe(df)

if choice == "Profile": 
    st.title("Profile")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Machine Learning": 
    target = st.selectbox("Choose the Target", df.columns)
    if st.button("Run Modelling"): 
        setup(df, target=target, silent=True)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        save_model(best_model, 'best_model')
        st.dataframe(compare_df)

if choice == "Download":
    with open("best_model.pkl", 'rb') as f: 
        st.download_button("Download Model", f, "best_model_test.pkl")
