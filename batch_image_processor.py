import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime

from make_text_embeddings import generate_text_embedding
from ollama_generate_description import generate_description

BATCH_SIZE = 1000

def get_image_embedding_function(model_type):
    if model_type.lower() == "clip":
        from make_image_embeddings import generate_image_embedding
    else:
        from make_image_embeddings_DINOv2 import generate_image_embedding
    return generate_image_embedding

def process_image(image_path, model_type):
    generate_image_embedding = get_image_embedding_function(model_type)
    image_embedding = generate_image_embedding(image_path=image_path)
    description = generate_description(image_path=image_path)
    description_embedding = generate_text_embedding(description)
    return image_embedding, description, description_embedding

def save_batch(batch_num, image_embeddings, desc_embeddings, image_paths, descriptions, output_dir):
    embed_file = output_dir / f"batch_{batch_num}_embeddings.npy"
    df_file = output_dir / f"batch_{batch_num}_data.csv"

    combined_embeddings = np.hstack([image_embeddings, desc_embeddings])
    np.save(embed_file, combined_embeddings)

    df = pd.DataFrame({
        "image_path": image_paths,
        "description": descriptions
    })
    df.to_csv(df_file, index=False)

    return embed_file, df_file

def combine_batches(output_dir):
    # Load and concatenate all embedding .npy files
    all_embeddings = []
    for file in sorted(output_dir.glob("batch_*_embeddings.npy")):
        emb = np.load(file)
        all_embeddings.append(emb)
    final_embeddings = np.vstack(all_embeddings)
    final_embedding_file = output_dir / "final_embeddings.npy"
    np.save(final_embedding_file, final_embeddings)

    # Load and concatenate all .csv files
    all_dataframes = [pd.read_csv(f) for f in sorted(output_dir.glob("batch_*_data.csv"))]
    final_df = pd.concat(all_dataframes, ignore_index=True)
    final_csv_file = output_dir / "final_data.csv"
    final_df.to_csv(final_csv_file, index=False)

    # Cleanup intermediate files
    for f in output_dir.glob("batch_*_embeddings.npy"):
        os.remove(f)
    for f in output_dir.glob("batch_*_data.csv"):
        os.remove(f)

    return final_embedding_file, final_csv_file

def process_images_in_batches(image_paths, output_dir, model_type):
    batch_num = 0
    image_embeddings = []
    desc_embeddings = []
    image_paths_batch = []
    descriptions = []

    for i, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
        try:
            img_emb, desc, desc_emb = process_image(image_path, model_type)
        except Exception as e:
            print(f"Skipping {image_path}: {e}")
            continue

        image_embeddings.append(img_emb)
        desc_embeddings.append(desc_emb)
        image_paths_batch.append(image_path)
        descriptions.append(desc)

        if len(image_embeddings) == BATCH_SIZE:
            save_batch(batch_num,
                       np.array(image_embeddings),
                       np.array(desc_embeddings),
                       image_paths_batch,
                       descriptions,
                       output_dir)
            batch_num += 1
            image_embeddings.clear()
            desc_embeddings.clear()
            image_paths_batch.clear()
            descriptions.clear()

    # Save final batch if it exists
    if image_embeddings:
        save_batch(batch_num,
                   np.array(image_embeddings),
                   np.array(desc_embeddings),
                   image_paths_batch,
                   descriptions,
                   output_dir)

    return combine_batches(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images to process.")
    parser.add_argument("--type", type=str, default="DINOv2", choices=["DINOv2", "CLIP", "clip"], help="Embedding model type: DINOv2 (default) or CLIP")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists() or not image_dir.is_dir():
        print("Provided image_dir is not a valid directory.")
        exit(1)

    # Collect image paths
    image_paths = sorted([str(p) for p in image_dir.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
    if not image_paths:
        print("No valid image files found in directory or subdirectories.")
        exit(1)

    # Create output directory named after input folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_name = f"{image_dir.stem}_{timestamp}"
    output_dir = Path("dataset") / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    final_embedding_file, final_csv_file = process_images_in_batches(image_paths, output_dir, args.type)

    print("\nAll batches processed.")
    print(f"Final embeddings saved to: {final_embedding_file}")
    print(f"Final CSV saved to: {final_csv_file}")
