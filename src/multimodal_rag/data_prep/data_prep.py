from multimodal_rag.tools.common import *
from multimodal_rag.common.config import get_settings
from multimodal_rag.tools.clip_encoder import *
from multimodal_rag.tools.milvus_db import create_collections, create_indexes, close_milvus_client, insert_data

settings = get_settings()

# Get chunked text content, and image content
processed_dir = settings.data_dir + "/" + settings.processed_subdir
text_collection_name = settings.text_collection_name
image_collection_name = settings.image_collection_name

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
    image_list.append(content['image_path'])


print(len(text_list))
print(len(image_list))

"""Compute embeddings using clip"""
text_embeddings = encode_text(text_list)
image_embeddings = encode_image(image_list)

print(text_embeddings.shape)
print(image_embeddings.shape)

# prepare data for text collection
text_data = []
for i, text_item in enumerate(text_content_list):
    single_embedding = text_embeddings[i].squeeze().tolist()
    text_item["vector"] = single_embedding
    text_data.append(text_item)

# prepare data for image collection
image_data = []
for i, image_item in enumerate(image_content_list):
    single_embedding = image_embeddings[i].squeeze().tolist()
    image_item["vector"] = single_embedding
    image_data.append(image_item)

print(len(text_data))
print(len(image_data))

# save_to_json(text_data, processed_dir + "/" + "text_data_milvus.json")
# save_to_json(image_data, processed_dir + "/" + "image_data_milvus.json")

# insert data into milvus
create_collections(text_collection_name=text_collection_name, image_collection_name=image_collection_name)
create_indexes(text_collection_name=text_collection_name, image_collection_name=image_collection_name)
insert_data(collection_name=text_collection_name, data=text_data)
insert_data(collection_name=image_collection_name, data=image_data)
close_milvus_client()
