from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch


### Generate image enbeddings using CLIP model 

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")


def generate_image_embedding(image_path):
    image = Image.open(image_path)

    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    return outputs.squeeze(0).numpy()