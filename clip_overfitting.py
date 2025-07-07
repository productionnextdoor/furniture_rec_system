import numpy as np
import faiss
import pandas as pd
import plotly.figure_factory as ff


#######################################################################################################
######## This file was used to chech whether CLIP embeddings are overfitting on the datsaset ##########
######## when comapred with DINOv2 embeddings the results can be seen in overfitting.txt ##############
######## the respective graph can be seen in graphs/overfitting #######################################
#######################################################################################################



def load_embeddings():
    image_embeddings_clip = np.load('clip_embeddings.npy')
    image_embeddings_dino = np.load('image_embeddings_DINOv2_combined.npy')
    return {
        "CLIP": image_embeddings_clip,
        "DINOv2": image_embeddings_dino,
    }

def normalize(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

embeddings_dict = load_embeddings()
normalized_embeddings_dict = {
    name: normalize(emb) for name, emb in embeddings_dict.items()
}


k = 5  # less so plotting, plot on rednering properly with 10 
indices = {}
score_idx = {}

for name, embeddings in normalized_embeddings_dict.items():
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    indices[name] = index

    distances, neighbors = index.search(embeddings, k)
    
    # Normalize distance and convert to similarity scores
    distances = (distances - distances.min()) / (distances.max() - distances.min())
    similarity = 1 - distances
    score_idx[name] = [similarity, neighbors]


similarity_clip, I_clip = score_idx["CLIP"]
similarity_dino, I_dino = score_idx["DINOv2"]

fig1 = ff.create_distplot(
    [similarity_clip.flatten(), similarity_dino.flatten()],
    ["CLIP", "DINOv2"],
    bin_size=0.02,
    show_hist=False,
    colors=["blue", "red"],
)
fig1.update_layout(
    title="Similarity Score Distribution: CLIP vs. DINOv2",
    xaxis_title="Similarity Score",
    yaxis_title="Density",
)
fig1.write_html("Similarity_Score_Distribution_CLIP_vs_DINOv2.html")


def compute_neighbor_diversity(I):
    unique_neighbors = [len(set(row)) for row in I]
    return np.mean(unique_neighbors) / k

clip_diversity = compute_neighbor_diversity(I_clip)
dino_diversity = compute_neighbor_diversity(I_dino)
print(f"Neighbor Diversity Ratio - CLIP: {clip_diversity:.2f}, DINOv2: {dino_diversity:.2f}")