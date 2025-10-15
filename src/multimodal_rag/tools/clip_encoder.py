import torch
import clip
from PIL import Image
from multimodal_rag.common.config import get_settings
from typing import List

# print(clip.available_models())

settings = get_settings()

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = settings.embedding_model
# print("embedding model: " + model_name)
model, preprocess = clip.load(model_name, device=device)
model.eval()

# Define a function to encode images
def encode_image(image_paths: List[str]):
    images = [preprocess(Image.open(path)) for path in image_paths]
    image_batch = torch.stack(images).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_batch)
        image_features /= image_features.norm(
            dim=-1, keepdim=True
        )  # Normalize the image features
    print(f"images的embedding形状：{image_features.shape}")
    return image_features

# Define a function to encode text
def encode_text(texts: List[str]):
    text_token_list = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_token_list)
        text_features /= text_features.norm(
            dim=-1, keepdim=True
        )  # Normalize the text features
    print(f"texts的embedding形状: {text_features.shape}")
    return text_features


# texts = ["test clip embedding", "this is an example", "a real world problem"]
# encode_text(texts)

# image_paths = ["/home/ubuntu/workspace/multimodal_rag/data/images/0*YE-Q-OuWnrgrUrQw.jpg", "/home/ubuntu/workspace/multimodal_rag/data/images/1*2X1aT8fzFsgbqn23zXmmAA.png"]
# encode_image(image_paths)

# texts = ["test clip embedding"]
# encode_text(texts)

# image_paths = ["/home/ubuntu/workspace/multimodal_rag/data/images/0*YE-Q-OuWnrgrUrQw.jpg"]
# encode_image(image_paths)
