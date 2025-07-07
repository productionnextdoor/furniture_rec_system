import torch
from PIL import Image
from torchvision import transforms


### Generate image enbeddings using DINOv2 model 

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval()  

# Preprocessing function to resize and pad image without cropping
def preprocess_image(image, target_size=(224, 224)):
    # Resize to fit the target size while maintaining aspect ratio
    aspect_ratio = image.width / image.height
    if aspect_ratio > 1:
        # Image is wide (landscape), so resize based on width
        new_width = target_size[0]
        new_height = int(target_size[0] / aspect_ratio)
    else:
        # Image is tall (portrait), so resize based on height
        new_height = target_size[1]
        new_width = int(target_size[1] * aspect_ratio)

    image = image.resize((new_width, new_height), Image.LANCZOS)

    # Create a new image with the target size, and paste the resized image in the center
    new_image = Image.new("RGB", target_size, (0, 0, 0))  # Black padding (you can choose white or any color)
    padding_left = (target_size[0] - new_width) // 2
    padding_top = (target_size[1] - new_height) // 2
    new_image.paste(image, (padding_left, padding_top))

    return new_image


def generate_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    
    image = preprocess_image(image)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():  
        image_features = model(image_tensor)

    return image_features.squeeze().numpy()
