import numpy as np
import pandas as pd
from make_image_embeddings_DINOv2 import generate_image_embedding


df = pd.read_csv('full_furniture_dataset_with_description.csv')  
df = df[df['Category'] != 'beds'].reset_index(drop=True)
df = df[['Image Path', 'Category']]

batch_size = 1000
current_batch = []
batch_number = 1  

def save_batch_embeddings(batch_embeddings, batch_number):
    batch_embeddings = np.array(batch_embeddings, dtype=np.float32)
    batch_filename = f'dataset/image_embeddings_DINOv2_batch_{batch_number}.npy'
    np.save(batch_filename, batch_embeddings)
    print(f"Batch {batch_number} embeddings saved successfully!")
    
for idx, row in df.iterrows():
    
    image_path = row['Image Path']
    image_embedding = generate_image_embedding(image_path)
    
    current_batch.append(image_embedding)

    if len(current_batch) >= batch_size:
        save_batch_embeddings(current_batch, batch_number)
        batch_number += 1  
        current_batch = []

if current_batch:
    save_batch_embeddings(current_batch, batch_number)

print("All embeddings processed and saved successfully!")
