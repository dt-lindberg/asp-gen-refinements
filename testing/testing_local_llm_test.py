"""Testing the testing script on local machine"""

import os

from transformers import pipeline
from transformers import AutoConfig, AutoProcessor
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    hf_token = os.getenv("HF_TOKEN", None)
    if hf_token is None:
        raise ValueError("HF_TOKEN not set/found in .env file")

    login(token=hf_token)

    pipe = pipeline("sentiment-analysis")
    print(pipe("hello from my laptop"))

    model_id = "Qwen/Qwen3.5-9B"

    config = AutoConfig.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    print(type(config).__name__)
    print(type(processor).__name__)
    print(config.model_type)
