from bs4 import BeautifulSoup
import os

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from torch import cat, save

from multimodal_rag.tools.common import *
from multimodal_rag.common.config import get_settings
from multimodal_rag.tools.clip_encoder import *

settings = get_settings()

# Get chunked text content, and image content
processed_dir = settings.data_dir + "/" + settings.processed_subdir
text_content_list = load_from_json(processed_dir + "/" + settings.text_content_file_name)
image_content_list = load_from_json(processed_dir + "/" + settings.image_content_file_name)

print(len(text_content_list))
print(len(image_content_list))

text_list = []
for content in text_content_list:
    # concatenate title and section header
    section = content['section'] + ": "
    # append text from paragraph to fill CLIP's 256 sequence limit
    text = section + content['text'][:256-len(section)]

    text_list.append(text)

image_list = []
for content in image_content_list:
    image_list.append(Image.open(content['image_path']))


print(len(text_list))
print(len(image_list))

"""Compute embeddings using clip"""
text_embeddings = encode_text(text_list)
image_embeddings = encode_image(image_list)

print(text_embeddings.shape)
print(image_embeddings.shape)
