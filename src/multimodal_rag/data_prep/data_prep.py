from bs4 import BeautifulSoup
import os

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from torch import cat, save

from multimodal_rag.tools.functions import *
from multimodal_rag.common.config import get_settings

settings = get_settings()

# Get all HTML files from raw directory
raw_dir = settings.data_dir + "/raw"
filename_list = [raw_dir+"/"+f for f in os.listdir(raw_dir)]

text_content_list = []
image_content_list = []
for filename in filename_list:

    with open(filename, 'r', encoding='utf-8') as file:
        html_content = file.read()

    text_content_list.extend(parse_html_content(html_content))
    image_content_list.extend(parse_html_images(html_content))

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

"""Compute embeddings using CLIP"""
# import model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

# import processor (handles text tokenization and image preprocessing)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# pre-process text and images
inputs = processor(text=text_list, images=image_list, return_tensors="pt", padding=True)

# compute embeddings with CLIP
outputs = model(**inputs)

# store embeddings in single torch tensor
text_embeddings = outputs.text_embeds
image_embeddings = outputs.image_embeds

print(text_embeddings.shape)
print(image_embeddings.shape)

# save content list as JSON
save_to_json(text_content_list, output_file='data/text_content.json')
save_to_json(image_content_list, output_file='data/image_content.json')

# save embeddings to file
save(text_embeddings, 'data/text_embeddings.pt')
save(image_embeddings, 'data/image_embeddings.pt')
