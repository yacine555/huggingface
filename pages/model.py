import streamlit as st
import pandas as pd
import numpy as np
from task import Task
import time

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Img2text", "Text2voiceM1", "SentimentAnalysis", "Summarizisation", "Img2Video", "K-Text2Img", "K-Img2Img"])

with tab1:
    st.header("Img2text")
    uploaded_file = st.file_uploader("Choose an image file...", type=["png", "jpg", "jpeg"])   

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as f:
            f.write(bytes_data)

        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        with st.spinner('Generating text...'):
            scenario = Task.img2text(uploaded_file.name)
            st.write(scenario)
            st.success('Done!')

with tab2:
    st.header("text2voiceM1")
    txt2 = st.text_area(
    "Text to analyze",
    "It was the best of times, it was the worst of times, it was the age of "
    "wisdom, it was the age of foolishness, it was the epoch of belief, it "
    "was the epoch of incredulity, it was the season of Light, it was the "
    "season of Darkness, it was the spring of hope, it was the winter of "
    "despair, (...)",key="txt")
    st.write(f'You wrote {len(txt2)} characters.')

    if st.button('Generate audio', key="B2"):
        with st.spinner('Wait for it...'):
            Task.story2voiceM1(txt2)

        st.success('Done!')
        st.audio("story.flac")


with tab3:
    st.header("SentimentAnalysis")
    txt3 = st.text_area(
    "Text to analyze",
    "It was the best of times, it was the worst of times, it was the age of "
    "wisdom, it was the age of foolishness, it was the epoch of belief, it "
    "was the epoch of incredulity, it was the season of Light, it was the "
    "season of Darkness, it was the spring of hope, it was the winter of "
    "despair, (...)",key="txt2")
    st.write(f'You wrote {len(txt3)} characters.')

    c31, c32 = st.columns([1,1])
    c31.empty()

    if st.button('Perform Sentiment Analysis', key="B3"):
        with st.spinner('Wait for it...'):
            c31.empty()
            time.sleep(2)
            res = Task.sentimentAnalysis(txt3)
            c31.write(res[0]) 
            c31.success('Done!')
    

with tab4:
    st.header("Summarizisation")
    txt4 = st.text_area(
    "Text to summarize",
    "Hugging Face Diffusers is the go-to library for state-of-the-art pretrained diffusion models for generating images, audio, and even 3D structures of molecules. Whether you’re looking for a simple inference solution or want to train your own diffusion model, 🤗 Diffusers is a modular toolbox that supports both. Our library is designed with a focus on usability over performance, simple over easy, and customizability over abstractions."
    "The library has three main components:"
    "State-of-the-art diffusion pipelines for inference with just a few lines of code. There are many pipelines in 🤗 Diffusers, check out the table in the pipeline overview for a complete list of available pipelines and the task they solve."
    "Interchangeable noise schedulers for balancing trade-offs between generation speed and quality."
    "Pretrained models that can be used as building blocks, and combined with schedulers, for creating your own end-to-end diffusion systems."
    "End",key="txt4")
    st.write(f'You wrote {len(txt4)} characters.')

    c41, c42 = st.columns([1,1])

    if st.button('Perform summarization', key="B4"):
        with st.spinner('Wait for it...'):
            c31.empty()
            time.sleep(2)
            res = Task.summarize(txt4)
            c41.write(res[0]) 
            c41.success('Done!')

with tab5:
    st.header("Img2Video")


    if st.button('Generate Video'):
        'We have that animal!'
        #Task.img2video()
    
with tab6:
    st.header("Kandinsky 3.0 text2img")

    if st.button('Generate image from prompt'):
        prompt = "A photograph of the inside of a subway train. There are raccoons sitting on the seats. One of them is reading a newspaper. The window shows the city in the background."
        img = Task.ktext2img(prompt)
        st.image(img, caption='Image generated', use_column_width=True)

with tab7:
    st.header("Kandinsky 3.0 img2img")

    if st.button('Generate image from image'):
        prompt = "A painting of the inside of a subway train with tiny raccoons."
        img = Task.kimg2img(prompt)
        st.image(img, caption='Image generated', use_column_width=True)