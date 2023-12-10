import requests
import os
import json
import sys,getopt
import config
import torch

from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import soundfile as sf

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI


from diffusers import StableVideoDiffusionPipeline
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, export_to_video

class Task:

    @staticmethod
    def img2text(imageUrl):
        """
        Higgingface task that convert an image to text

        :param imageUrl: name of the image to describe (string)
        :return: text describing the image (string)
        """

        # Use a pipeline as a high-level helper
        image_to_textPipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
        text = image_to_textPipe(imageUrl)[0]['generated_text']
        return text
    
    @staticmethod
    def generateStory(scenario, modelname="gpt-3.5-turbo",temp=1):

        """
        Huggingface task to generate a story based on a text scenario
        Use Longchain prompt template with OpenAI LLM

        :param scenario: text scenario to generate story (string)
        :return: text story (string)
        """
        print("Exemple -  Call LLM with a Lonchain prompt tempate: \n")

        promptTempalte = """
        You are a story teller;
        You can generate a short story based on a simple narrative, the story should be no more than 100 words;
        
        CONTEXT: {scenario}
        STORY:
        """


        prompt = PromptTemplate(
            input_variables=["scenario"],
            template=promptTempalte,
        )   

        human_template = "{scenario}"

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", promptTempalte),
            ("human", human_template),
        ])


        story_llm = LLMChain(
            llm = ChatOpenAI( model_name=modelname, temperature=temp),
            prompt = chat_prompt,
            verbose = True)

        story = story_llm.predict(scenario=scenario)

        return story


    @staticmethod
    def story2voiceM1(story):
        """
        Huggingface task to generate voice based on a story. 
        Methode1 Using HF inference API model

        :param story: text to be converted to audio (string)
        :return: store story.flac in the local drve story (flac file)
        """

        API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
        headers = {"Authorization": f"Bearer {config.huggingfapi_key}"}

        print("key: " + config.huggingfapi_key)
        print("Hearder: " + json.dumps(headers))

        payload = {
            "inputs": story,
        }

        response = requests.post(API_URL, headers=headers, json=payload)

        #if response.status_code == 200:
        print(response.status_code)
        print(response.reason)

        folder_path = "output/"
        isExist = os.path.exists(folder_path)
        if not isExist:
            os.makedirs(folder_path)

        print("Generate audio file")
        with open(folder_path + "story.flac", "wb") as f:
            f.write(response.content)



    @staticmethod
    def story2voiceM2(story):
        """
        Huggingface task to generate voice based on a story. 
        Methode 2 Using downloaded model

        :param story: text to be converted to audio (string)
        :return: create story.flac in the local file story (audio flac file)
        """
        # Use a pipeline as a high-level helper
        taskPipe = pipeline("text-to-speech", model="espnet/kan-bayashi_ljspeech_vits")

        res= taskPipe(story)
        print("Generate audio file: " + res)
        
        folder_path = "output/"
        isExist = os.path.exists(folder_path)
        if not isExist:
            os.makedirs(folder_path)

        with open(folder_path + "story.flac", "wb") as f:
            f.write(res.content)

        return 
    

    @staticmethod
    def story2voiceM3(story):
        """
        Huggingface task to generate voice based on a story using the Microsoft model
        Methode 2 Using downloaded model

        :param story: text to be converted to audio (string)
        :return: create story.wav file in the local file (audio wav file)
        """
        # Use a pipeline as a high-level helper
        synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        # You can replace this embedding with your own as well.

        speech = synthesiser(story, forward_params={"speaker_embeddings": speaker_embedding})

        folder_path = "output/"
        isExist = os.path.exists(folder_path)
        if not isExist:
            os.makedirs(folder_path)

        sf.write(folder_path + "story.wav", speech["audio"], samplerate=speech["sampling_rate"])

        return 
    

    @staticmethod
    def sentimentAnalysis(text):
        """
        Huggingface task to do sentiment analysis using Huggingface transformers pipeline

        :param story: text to be converted to audio (string)
        :return: [{'label': 'NEGATIVE|NEUTRAL|POSITIVE', 'score': 0 to 1}] (json object )
        """
        classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        res = classifier(text)
        return res

    @staticmethod
    def sentimentAnalysisTokenizer(text):
        """
        Huggingface task to do sentiment analysis with tokenizers using Huggingface transformers pipeline

        :param story: text to analysis (string)
        :return: [{'label': 'NEGATIVE|NEUTRAL|POSITIVE', 'score': 0 to 1}] (json object )
        """
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        res = classifier(text)
        return res
    

    @staticmethod
    def sentimentAnalysisTokenizerTest(text):
        """
        Huggingface task to do sentiment analysis with tokenizers using Huggingface transformers pipeline with print of tokens data

        :param story: text to be tokenize (string)
        :return: None
        """
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
        res = tokenizer(text)
        print(res)

        tokens = tokenizer.tokenize(text)
        print(tokens)

        ids = tokenizer.convert_tokens_to_ids(tokens)
        print(ids)

        decoded_string = tokenizer.decode(ids)
        print(decoded_string)



    @staticmethod
    def summarize(text):
        """
        Huggingface task to summarize text 

        :param text: text to be summarize (string)
        :return: (string)
        """
        summarizerTask = pipeline("summarization", model="facebook/bart-large-cnn")
        res = summarizerTask(text,max_length=150,min_length=100, do_sample=False)
        return res


    @staticmethod
    def img2video():
        """
        Huggingface task to convert an image to video using stability AI model
        """
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", 
            torch_dtype=torch.float32, 
            variant="fp16"
        )
    
        #pipe.enable_model_cpu_offload()

        image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
        image = image.resize((512,288))

        generator = torch.manual_seed(42)
        frames = pipe(image, decode_chunk_size=1, generator=generator).frames[0]

        export_to_video(frames, "generated.mp4", fps=7)

    @staticmethod
    def ktext2img(prompt_text):
        """
        Huggingface diffuser. Kandinsky 3.0 convert text to imgage task using K-diffusion model
        """

        pipe = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float32)
        #pipe.enable_model_cpu_offload()
                
        prompt = "A photograph of the inside of a subway train. There are raccoons sitting on the seats. One of them is reading a newspaper. The window shows the city in the background."

        generator = torch.Generator(device="cpu").manual_seed(0)
        image = pipe(prompt, num_inference_steps=25, generator=generator).images[0]

        return image


    @staticmethod
    def kimg2img(prompt_text):
        """
        Huggingface diffuser. Kandinsky 3.0 convert image to imgage task using K-diffusion model
        """

        pipe = AutoPipelineForImage2Image.from_pretrained("kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float32)
        #pipe.enable_model_cpu_offload()
                
        prompt = "A painting of the inside of a subway train with tiny raccoons."
        image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky3/t2i.png")

        generator = torch.Generator(device="cpu").manual_seed(0)
        image = pipe(prompt, image=image, strength=0.75, num_inference_steps=25, generator=generator).images[0]

        return image