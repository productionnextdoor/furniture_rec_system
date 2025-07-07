import argparse
from pathlib import Path
from make_text_embeddings import generate_text_embedding
from ollama_generate_description import generate_description

def get_image_embedding_function(model_type):
    if model_type.lower() == "clip":
        from make_image_embeddings import generate_image_embedding
    else:
        from make_image_embeddings_DINOv2 import generate_image_embedding
    return generate_image_embedding

def process_image(image_path, model_type="DINOv2"):
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    generate_image_embedding = get_image_embedding_function(model_type)
    image_embedding = generate_image_embedding(image_path=image_path)
    description = generate_description(image_path=image_path)
    description_embedding = generate_text_embedding(description)
    
    return image_embedding, description, description_embedding

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--type", type=str, default="DINOv2", choices=["DINOv2", "CLIP", "clip"], help="Embedding model type: DINOv2 (default) or CLIP")
    args = parser.parse_args()

    emb, desc, desc_emb = process_image(args.image_path, args.type)
    print("Image embedding:", emb)
    print("Description:", desc)
    print("Description embedding:", desc_emb)
