import numpy as np
import faiss
import pandas as pd
from tqdm import tqdm
import os


THRESHOLD = 0.99  
CLIP_EMBEDDINGS_PATH = "dataset/clip_embeddings.npy"
WHOLE_EMBEDDINGS_PATH = "dataset/embeddings.npy"
METADATA_EMBEDDINGS_PATH = "dataset/metadata_embeddings.npy"
TEXT_EMBEDDINGS_PATH = "dataset/text_embeddings.npy"
CSV_PATH = "dataset/clip_metadata_desc_embed.csv"
DINO_EMBEDDINGS_PATH = "dataset/image_embeddings_DINOv2_combined.npy"

CLIP_OUTPUT_EMBEDDINGS = "dataset/clip_embeddings_dedup.npy"
WHOLE_EMBEDDINGS_OUTPUT_PATH = "dataset/embeddings_dedup.npy"
METADATA_EMBEDDINGS_OUTPUT_PATH = "dataset/metadata_embeddings_dedup.npy"
TEXT_EMBEDDINGS_OUTPUT_PATH = "dataset/text_embeddings_dedup.npy"
OUTPUT_CSV = "dataset/clip_metadata_desc_embed_dedup.csv"
DINO_EMBEDDINGS_OUPUT_PATH = "dataset/image_embeddings_DINOv2_combined_dedup.npy"


print("Loading embeddings...")
embeddings = np.load(CLIP_EMBEDDINGS_PATH).astype('float32')
whole_embeddings = np.load(WHOLE_EMBEDDINGS_PATH).astype('float32')
metadata_embeddings = np.load(METADATA_EMBEDDINGS_PATH).astype('float32')
text_embedddings  = np.load(TEXT_EMBEDDINGS_PATH).astype('float32')
dino_embeddings = np.load(DINO_EMBEDDINGS_PATH).astype('float32')
df = pd.read_csv(CSV_PATH)  


print("Finding duplicates...")
dim = embeddings.shape[1]

# Normalize embeddings (for cosine similarity)
faiss.normalize_L2(embeddings)

# Build FAISS index
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# Search for nearest neighbors (k=2: self + closest)
k = 2
D, I = index.search(embeddings, k)


duplicates = set()
for i in tqdm(range(len(embeddings))):
    if i not in duplicates:
        for j in I[i][1:]:  
            if D[i][1] > THRESHOLD:
                duplicates.add(j)


print("Removing duplicates...")
non_duplicates = [i for i in range(len(embeddings)) if i not in duplicates]
with open("dataset/nonduplicate.txt", 'a') as f:
    f.write(", ".join(str(item) for item in non_duplicates))

df_dedup = df.iloc[non_duplicates].reset_index(drop=True)


embeddings_dedup = embeddings[non_duplicates]
whole_embeddings_dedup = whole_embeddings[non_duplicates]
metadata_embedding_dedup = metadata_embeddings[non_duplicates]
text_embedddings_dedup = text_embedddings[non_duplicates]
dino_embeddings_dedup = dino_embeddings[non_duplicates]

print("Deleting duplicate files...")
for idx in tqdm(duplicates):
    img_path = df.loc[idx, 'Image Path']
    if os.path.exists(img_path):
        os.remove(img_path)
    else:
        print(f"Warning: File not found (skipping): {img_path}")


print("Saving results...")
np.save(CLIP_OUTPUT_EMBEDDINGS, embeddings_dedup)
np.save(WHOLE_EMBEDDINGS_OUTPUT_PATH, whole_embeddings_dedup)
np.save(METADATA_EMBEDDINGS_OUTPUT_PATH, metadata_embedding_dedup)
np.save(TEXT_EMBEDDINGS_OUTPUT_PATH, text_embedddings_dedup)
np.save(DINO_EMBEDDINGS_OUPUT_PATH, dino_embeddings_dedup)
df_dedup.to_csv(OUTPUT_CSV, index=False)


print("\n=== Deduplication Complete ===")
print(f"Original size: {len(df)} images")
print(f"New size: {len(df_dedup)} images")
print(f"Duplicates removed: {len(duplicates)}")
print(f"Cleaned embeddings saved to: {CLIP_OUTPUT_EMBEDDINGS}")
print(f"Cleaned DataFrame saved to: {OUTPUT_CSV}")