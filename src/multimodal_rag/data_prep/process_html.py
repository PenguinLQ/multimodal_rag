import os
from multimodal_rag.tools.common import *
from multimodal_rag.common.config import get_settings

settings = get_settings()

# Get all HTML files from raw directory
raw_dir = settings.data_dir + "/" + settings.raw_subdir
filename_list = [raw_dir+"/"+f for f in os.listdir(raw_dir)]
processed_dir = settings.data_dir + "/" + settings.processed_subdir

text_content_list = []
image_content_list = []
for filename in filename_list:

    with open(filename, 'r', encoding='utf-8') as file:
        html_content = file.read()

    text_content_list.extend(parse_html_content(html_content))
    image_content_list.extend(parse_html_images(html_content))

print(len(text_content_list))
print(len(image_content_list))

# save content list as JSON
save_to_json(text_content_list, output_file=processed_dir+'/text_content.json')
save_to_json(image_content_list, output_file=processed_dir+'/image_content.json')
