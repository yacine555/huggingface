import requests
import config
import os
import json
import sys,getopt
import pandas as pd
import numpy as np
from dotenv import load_dotenv, find_dotenv

from task import Task

import streamlit as st
import subprocess


load_dotenv(find_dotenv())
config.openapi_key=os.getenv('OPENAI_API_KEY')
config.huggingfapi_key=os.getenv('HUGGINGFACEHUB_API_TOCKEN')

st.markdown("# Main")
st.sidebar.markdown("# Main")

def runstreamlit():

    #st.set_page_config(layout="wide" , page_title="Img 2 audio story", page_icon = "🚀")
    st.title("Turn imgage into audio story")

    # Add a selectbox to the sidebar:
    llm_selectbox = st.sidebar.selectbox(
        'Select the OpenAI LLM model:',
        ('gpt-3.5-turbo', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106')
    )


    temp_slider= st.sidebar.slider(label="Temperature", min_value=0.0, max_value=1.0, value=1.0, step=0.1)

    uploaded_file = st.file_uploader("Choose an image file...", type=["png", "jpg", "jpeg"])   

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as f:
            f.write(bytes_data)

        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        scenario = Task.img2text(uploaded_file.name)
        story = Task.generateStory(scenario,modelname=llm_selectbox,temp=temp_slider)
        Task.story2voiceM1(story)

        with st.expander("Scenario"):
            st.write(scenario)
        with st.expander("Story"):
            st.write(story)
        st.audio("story.flac")


def main(argv):
    print ('Run main ')
    taskNum = ''
    streamlit = 'NO'
    opts, args = getopt.getopt(argv,"ht:",["task="])
    print ('opts ', opts)

    runstreamlit()



if __name__ == "__main__":
    main(sys.argv[1:])
