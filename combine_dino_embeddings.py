import numpy as np
import os

batch_files_path = 'dataset/' 

batch_files = [f for f in os.listdir(batch_files_path) if f.startswith('image_embeddings_DINOv2_batch_') and f.endswith('.npy')]

batch_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

all_embeddings = []

for batch_file in batch_files:
    batch_embeddings = np.load(os.path.join(batch_files_path, batch_file))
    all_embeddings.append(batch_embeddings)
    os.remove(batch_file)

all_embeddings = np.concatenate(all_embeddings, axis=0)

np.save('dataset/image_embeddings_DINOv2_combined.npy', all_embeddings)



print("All batch embeddings combined and saved successfully as 'image_embeddings_DINOv2_combined.npy'!")
