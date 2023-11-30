# Huggingface Face - with langchain 

## Description

This project explains how to use Huggingface locally by creating a VisionVoice App which turns an image to a story through a 3 part process

Part 1: image to text
Part 2: Use an LLM model to provide a story
Part 3: Read the story with voice

It also contains some addional Hugging Face tasks execution for model testing and discovery purpose

## Getting Started

Follow these instructions to get the VisionVoice app running on your local machine for development and testing purposes.

### Prerequisites and Dependencies

Before you begin, ensure you have the following installed:
- Python 3.10.10 or later. Note that this was only tested on 3.10.10
- [Pytorch deeplearning library](https://pytorch.org/get-started/locally/)
- [Langchain 0.0.331](https://python.langchain.com/docs/get_started/introduction)
- [Streamlit](https://streamlit.io/) 
- Check Huggingface installation resources [page](https://huggingface.co/docs/transformers/installation)


Here are the PIP module used

- [**python-dotenv (1.0.0)**](https://pypi.org/project/python-dotenv/1.0.0/): Reads key-value pairs from a `.env` file and sets them as environment variables.
- [**transformers (4.35.2)**](https://pypi.org/project/transformers/4.35.2/): Provides state-of-the-art general-purpose architectures for NLP, including BERT, GPT-2, T5, and others.
- [**datasets (2.15.0)**](https://pypi.org/project/datasets/2.15.0/): Offers a large collection of ready-to-use datasets for NLP model training and evaluation.
- [**langchain (0.0.341)**](https://pypi.org/project/langchain/0.0.341/): Designed for building applications involving language models.
- [**streamlit (1.28.2)**](https://pypi.org/project/streamlit/1.28.2/): An app framework for Machine Learning and Data Science to create apps quickly.
- [**config (0.5.1)**](https://pypi.org/project/config/0.5.1/): Handles configuration files in Python.
- [**soundfile (0.12.1)**](https://pypi.org/project/SoundFile/0.12.1/): Reads from and writes to various sound file formats, often used with NumPy.


### Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/image2voicehflc-app.git
cd image2voicehflc-app
pip install -r requirements.txt
```



### Setting Up

To use Hugging Face and Langchain services, you need to sign up for their APIs and set your API keys:

```bash
export HUGGINGFACE_API_KEY='your_huggingface_api_key'
export LANGCHAIN_API_KEY='your_langchain_api_key'
```


### Running the Application

Start the application by running:

```bash
python app.py -t 1 
```
or

```bash
python app.py --task 1 
```

Run the the app streamlit
```bash
streamlit run myApp.py
```



## Deployment

Notes on how to deploy the application in a live environment.

## Built With

- [Framework](#) - The web framework used.
- [Database](#) - Database system.
- [Others](#) - Any other frameworks, libraries, or tools used.

## Contributing

Guidelines for contributing to the project.

[CONTRIBUTING.md](CONTRIBUTING.md)

## Versioning

Information on versioning system used, typically [SemVer](http://semver.org/).

## Authors

- **Name** - *Initial work* - [Profile](#)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Acknowledgments to individuals or projects that assisted in development.
