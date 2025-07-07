import streamlit as st
import numpy as np
import pandas as pd
import faiss
from PIL import Image


###########################################################################################
######## This file was used to render the streamlit app ###################################
###########################################################################################



embeddings_matrix = np.load('dataset/main_dino_demo_embeddings.npy')
demo_embedding  = np.load('dataset/dino_demo_embeddings.npy')
df = pd.read_csv('dataset/clip_metadata_desc_embed_dedup.csv')
demo_df = pd.read_csv('dataset/dino_demo_dataset.csv')

normalized_embeddings_matrix = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)

def show_image_with_metadata(image_path, metadata, similarity_score=None, idx=None):
    """
    Helper function to display the image and metadata.
    """
    image = Image.open(image_path)
    col1, col2 = st.columns([1, 3])  # Create two columns for the image and metadata

    with col1:
       st.image(image, caption=f"Image {idx}", use_column_width=True)

    with col2:
        st.write("**Description**:")
        st.write(metadata)
        if similarity_score is not None:
            st.write(f"**Distance from Query**: {similarity_score:.4f}")


def manhattan_distance(A, B):
    """Manhattan (L1) distance"""
    return np.sum(np.abs(A - B), axis=1)

def chebyshev_distance(A, B):
    """Chebyshev (L∞) distance"""
    return np.max(np.abs(A - B), axis=1)


def test_query(query_idx, k=5, distance_type="L2"):
    """
    Perform a similarity search and display results with images and metadata for the query and its k nearest neighbors.
    """
    query_embedding = demo_embedding[query_idx]
    if distance_type == "L2":
        index = faiss.IndexFlatL2(embeddings_matrix.shape[1])
        index.add(embeddings_matrix.astype(np.float32))  
        D, I = index.search(np.array([query_embedding]), k)
    elif distance_type == "Cosine":
        query_embedding = query_embedding / np.linalg.norm(query_embedding)  
        index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
        index.add(normalized_embeddings_matrix.astype(np.float32))  
        D, I = index.search(np.array([query_embedding]), k)
        D = 1 - D       # D is similaity score as embeddings were normalized 
    elif distance_type == "Manhattan":
        distances = manhattan_distance(query_embedding, embeddings_matrix)
        distances_sorted_idx = np.argsort(distances)
        D = distances[distances_sorted_idx[:k]]
        I = distances_sorted_idx[:k]
        I = I.reshape(1, -1)  
    elif distance_type == "Chebyshev":
        distances = chebyshev_distance(query_embedding, embeddings_matrix)
        distances_sorted_idx = np.argsort(distances)
        D = distances[distances_sorted_idx[:k]]
        I = distances_sorted_idx[:k]
        I = I.reshape(1, -1)  

    st.write(f"### Query Image (Index: {query_idx}) and its Metadata using {distance_type} distance")
    query_row = demo_df.iloc[query_idx]
    image_path = query_row['Image Path']
    metadata = query_row['Description']
    if distance_type in ('L2', 'Cosine'):
        show_image_with_metadata(image_path, metadata, similarity_score=0.0000, idx=query_idx)
        st.divider()
    else:
        show_image_with_metadata(image_path, metadata, similarity_score=0.0000, idx=query_idx)
        st.divider()

    st.write(f"### {k} Most Similar Items:")
    for i, idx in enumerate(I[0]):
        if idx == query_idx:
            continue  
        row = df.iloc[idx]
        image_path = row['Image Path']
        metadata = row['Generated Description']
        if distance_type in ('L2', 'Cosine'):
            show_image_with_metadata(image_path, metadata, similarity_score=D[0][i], idx=idx)
            st.divider()
        else:
            show_image_with_metadata(image_path, metadata, similarity_score=D[i], idx=idx)
            st.divider()


st.set_page_config(page_title="Furniture Similarity Search", layout="wide")

st.markdown("# 🔍 **Furniture Similarity Search**")
st.markdown(
    "Welcome to the **Furniture Similarity Search Tool**. This tool allows you to find similar furniture items from a dataset based on their features and visual appearance. Select a distance metric to search for items similar to the one you choose!"
)

col1, col2 = st.columns([3, 1])
with col1:
    query_idx_input = st.text_input('Enter Image Index for Query:', '0')  
with col2:
    k_input = st.text_input('Enter Number of Nearest Neighbors:', '4')
    k_input = str(int(k_input))                                       

distance_type = st.selectbox(
    'Select Distance Measure:',
    [
        'L2', 'Cosine', 'Manhattan', 'Chebyshev'
    ],
    index=0
)

if query_idx_input.isdigit() and k_input.isdigit():
    query_idx = int(query_idx_input)
    k = int(k_input)

    if 0 <= query_idx < len(demo_df):
        if st.button("🔍 Search Similar Items"):
            with st.spinner('Searching for similar items...'):
                test_query(query_idx, k, distance_type)
    else:
        st.error(f"Index {query_idx} is out of bounds. Please enter a valid index (0 to {len(demo_df)-1}).")
else:
    st.error("Please enter valid numeric values for both Image Index and Number of Neighbors.")
