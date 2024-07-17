import os

from dotenv import load_dotenv


load_dotenv()

llava_parameters = {
    "prompt": "Express the image in one sentence with a facial expression.",
    "model_path": "liuhaotian/llava-v1.5-13b",
    "model_base": None,
    "device": "cuda:0",
    "conv_mode": None,
    "temperature": 0.2,
    "max_new_tokens": 128,
    "load_8bit": False,
    "load_4bit": True,
    "debug": False,
    "image_aspect_ratio": "pad"
}

deepl_parameters = {
    "deepl_key": os.getenv('DEEPL_KEY')
}