import numpy as np
import pandas as pd
from make_image_embeddings import generate_image_embedding
from make_text_embeddings import generate_text_embedding

# Load dataset
df = pd.read_csv('dataset/full_furniture_dataset_with_description.csv')  
df = df[df['Category'] != 'beds'].reset_index(drop=True)

# Initialize empty lists for different embeddings
image_embeddings = []
metadata_embeddings = []
text_embeddings = []

# Generate embeddings separately
for idx, row in df.iterrows():
    if not idx % 100:
        print(f"Generating embeddings for row no. {idx+1}")

    image_path = row['Image Path']
    metadata = {col: row[col] for col in df.columns if col not in ['Image Path', 'Generated Description']}
    description = row['Generated Description']

    # Generate image embedding
    image_embedding = generate_image_embedding(image_path)
    image_embeddings.append(image_embedding)

    # Generate metadata embedding (converting structured metadata to text)
    metadata_text = " ".join([f"{key}: {value}" for key, value in metadata.items()])
    metadata_embedding = generate_text_embedding(metadata_text)
    metadata_embeddings.append(metadata_embedding)
    
    # Generate text description embedding
    text_embedding = generate_text_embedding(description)
    text_embeddings.append(text_embedding)

# Convert lists to numpy arrays
image_embeddings = np.vstack(image_embeddings)
image_embeddings = np.array(image_embeddings, dtype=np.float32)
metadata_embeddings = np.vstack(metadata_embeddings)
metadata_embeddings = np.array(metadata_embeddings, dtype=np.float32)
text_embeddings = np.vstack(text_embeddings)
text_embeddings = np.array(text_embeddings, dtype=np.float32)

# Save embeddings for later use
np.save('clip_embeddings.npy', image_embeddings)
np.save('metadata_embeddings.npy', metadata_embeddings)
np.save('text_embeddings.npy', text_embeddings)

print("Embeddings generated and saved successfully!")
