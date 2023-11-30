import streamlit as st
import pandas as pd
import numpy as np
from task import Task

st.markdown("# Test Huggingface model")
st.sidebar.markdown("#  Parameters Huggingface model")

df = pd.DataFrame({
'first column': [1, 2, 3, 4, 5, 6],
'second column': [10, 20, 30, 40,50, 60 ]
})

option = st.sidebar.selectbox(
    'Which task model would you like to test?',
    df['first column'])

st.sidebar.markdown("You selected: ")
st.sidebar.write(option)

def cllbckSA():
    res = Task.sentimentAnalysis(txt2)
    c1.write(res[0]) 


tab1, tab2, tab3 = st.tabs(["Img2text", "Text2voiceM1", "SentimentAnalysis"])

with tab1:
    st.header("Img2text")
    uploaded_file = st.file_uploader("Choose an image file...", type=["png", "jpg", "jpeg"])   

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as f:
            f.write(bytes_data)

        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        scenario = Task.img2text(uploaded_file.name)
        st.write(scenario)

with tab2:
    st.header("text2voiceM1")
    txt = st.text_area(
    "Text to analyze",
    "It was the best of times, it was the worst of times, it was the age of "
    "wisdom, it was the age of foolishness, it was the epoch of belief, it "
    "was the epoch of incredulity, it was the season of Light, it was the "
    "season of Darkness, it was the spring of hope, it was the winter of "
    "despair, (...)",key="txt")
    st.write(f'You wrote {len(txt)} characters.')

    st.button(label="Generate audio", key=None, help=None, on_click=Task.story2voiceM1(txt), args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False)

with tab3:
    st.header("SentimentAnalysis")
    txt2 = st.text_area(
    "Text to analyze",
    "It was the best of times, it was the worst of times, it was the age of "
    "wisdom, it was the age of foolishness, it was the epoch of belief, it "
    "was the epoch of incredulity, it was the season of Light, it was the "
    "season of Darkness, it was the spring of hope, it was the winter of "
    "despair, (...)",key="txt2")
    st.write(f'You wrote {len(txt2)} characters.')

    st.button(label="Perform Sentiment Analysis", key=None, help=None, on_click=cllbckSA, args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False)
    c1, c2 = st.columns([1,1])

