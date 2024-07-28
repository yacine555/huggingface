import requests
import config
import os
import json
import sys, getopt
import pandas as pd
import numpy as np

from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from task import Task

import streamlit as st
import subprocess

load_dotenv(find_dotenv())


config.openapi_key = os.getenv("OPENAI_API_KEY")
config.huggingfapi_key = os.getenv("HUGGINGFACEHUB_API_TOCKEN")


def main(argv):
    taskNum = ""
    opts, args = getopt.getopt(argv, "ht:", ["task="])
    print("Main function Args opts:  ", opts)

    if len(sys.argv) > 1:
        for opt, arg in opts:
            if opt == "-h":
                print("test.py -t <tasknumber>")
                sys.exit()
            elif opt in ("-t", "--task"):
                taskNum = arg
                print("Run task ", taskNum)
                match taskNum:
                    case "1":
                        scenario = Task.img2text("photo1.png")
                        print("Task 1 - img2text:" + scenario + "  \n\n")
                    case "2":
                        scenario = Task.img2text("photo1.png")
                        print("Task 1 - img2text:" + scenario + "  \n\n")
                        story = Task.generateStory(scenario)
                        print("Task 2 - generateStory:" + story + "  \n\n")

                    case "3":
                        # scenario = Task.img2text("photo1.png")
                        # story = Task.generateStory(scenario)
                        # Task.story2voiceM1(story)
                        # Task.story2voiceM2("Helo World")
                        Task.story2voiceM3("Helo World")
                    case "4":
                        sentiment = Task.sentimentAnalysis(
                            "dear Z-mobile your service is not great in dallas..."
                        )
                        print("Task Sentiment - sentimentAnalysis:  \n\n")
                        print(sentiment)
                    case "5":
                        print("Task Sentiment - sentimentAnalysis Test token:  \n\n")
                        sentiment = Task.sentimentAnalysisTokenizerTest(
                            "dear Z-mobile your service is not great in dallas.."
                        )
                    case "6":
                        print("Task - Summarization \n\n")
                        text = """
                        Humane has been teasing its first device, the AI Pin, for most of this year. It's scheduled to launch the Pin on Thursday, but The Verge has obtained documents detailing practically everything about the device ahead of its official launch. What they show is that Humane, the company noisily promoting a world after smartphones, is about to launch what amounts to a $699 wearable smartphone without a screen that has a $24-a-month subscription fee and runs on a Humane-branded version of T-Mobile's network with access to AI models from Microsoft and OpenAI.
                        The Pin itself is a square device that magnetically clips to your clothes or other surfaces. The clip is more than just a magnet, though; it's also a battery pack, which means you can swap in new batteries throughout the day to keep the Pin running. We don't know how long a single battery lasts, but the device ships with two “battery boosters.” It's powered by a Qualcomm Snapdragon processor and uses a camera, depth, and motion sensors to track and record its surroundings. It has a built-in speaker, which Humane calls a “personic speaker,” and can connect to Bluetooth headphones. 
                        Since there's no screen, Humane has come up with new ways to interact with the Pin. It's primarily meant to be a voice-based device, but there's also that green laser projector we've seen in demos, which can project information onto your hand. You can also hold objects up to the camera and interact with the Pin through gestures, as there's a touchpad somewhere on the device. The Pin isn't always recording or even listening for a wake word, instead requiring you to manually activate it in some way. It has a “Trust Light,” which blinks on whenever the Pin is recording.
                        """
                        res = Task.summarize(text)
                        print(res)

                    case _:
                        print("Run default task ", taskNum)
                        subprocess.run(["ls", "-l"])


if __name__ == "__main__":
    main(sys.argv[1:])
