from pathlib import Path
import numpy as np
import pandas as pd
from ollama_generate_description import generate_description
from make_image_embeddings import generate_image_embedding
from make_text_embeddings import generate_text_embedding
from create_datasets import combine_embeddings


###########################################################################################
######## This file was used to generate the demo Datasets #################################
###########################################################################################

image_embeddings = np.load("dataset/clip_embeddings_dedup.npy")
description_embeddings = np.load('dataset/text_embeddings_dedup.npy')

main_demo_embeddings = np.array(combine_embeddings(image_embeddings, description_embeddings), dtype=np.float32)

np.save('dataset/main_demo_embeddings.npy', main_demo_embeddings)

############## Creating new small dataset for demo ##############
print("Working on the new dataset.")

demo_image_dir = 'dataset/dataset6/data'
image_paths = list(Path('dataset/dataset6/data').glob('*.png'))
demo_img_embeddings = []
demo_dsc_embeddings = []
demo_dataset = []

for i, image_full_path in enumerate(image_paths):
    if not i % 10:
        print(f"Processing file number {i}.")
    image_embedding = generate_image_embedding(image_path=image_full_path)
    description = generate_description(image_path=image_full_path)
    description_embedding = generate_text_embedding(description)

    demo_dataset.append({
        'Image Path': image_full_path,
        'Description': description
    })
    
    demo_img_embeddings.append(image_embedding)
    demo_dsc_embeddings.append(description_embedding)


df = pd.DataFrame(demo_dataset)
df.to_csv('dataset/demo_dataset.csv', index=False)
print("Dataset created successfully!")


demo_img_embeddings = np.array(np.vstack(demo_img_embeddings), dtype=np.float32)
demo_dsc_embeddings = np.array(np.vstack(demo_dsc_embeddings), dtype=np.float32)
demo_embeddings = np.array(combine_embeddings(demo_img_embeddings, demo_dsc_embeddings), dtype=np.float32)
np.save('dataset/demo_embeddings.npy', demo_embeddings)

print("Demo Embeddings created")



