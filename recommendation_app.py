import argparse
import numpy as np
import pandas as pd
import streamlit as st
import faiss
from PIL import Image
import os

from make_text_embeddings import generate_text_embedding
from ollama_generate_description import generate_description

###########################################################################################
######## This file was used to render the streamlit app ###################################
###########################################################################################

st.set_page_config(page_title="Furniture Similarity Search", layout="wide")

@st.cache_resource
def load_data(embedding_path, metadata_path):
    embeddings = np.load(embedding_path)
    df = pd.read_csv(metadata_path)
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings, normalized, df

def get_image_embedding_function(model_type):
    if model_type.lower() == "clip":
        from make_image_embeddings import generate_image_embedding
    else:
        from make_image_embeddings_DINOv2 import generate_image_embedding
    return generate_image_embedding

def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_path", type=str, required=True, help="Path to .npy embedding file")
    parser.add_argument("--dataframe_path", type=str, required=True, help="Path to .csv dataframe file")
    parser.add_argument("--type", type=str, default="DINOv2", choices=["DINOv2", "CLIP", "clip"], help="Embedding model type: DINOv2 (default) or CLIP")
    args, _ = parser.parse_known_args()  # allow Streamlit to ignore unknown args
    return args.embedding_path, args.dataframe_path, args.type

def manhattan_distance(A, B):
    return np.sum(np.abs(A - B), axis=1)

def chebyshev_distance(A, B):
    return np.max(np.abs(A - B), axis=1)

def show_image_with_metadata(image_path, metadata, similarity_score=None, idx=None):
    image = Image.open(image_path)
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(image, caption=f"Image {idx}", use_column_width=True)
    with col2:
        st.write("**Metadata**:")
        for key, value in metadata.items():
            st.write(f"{key}: {value}")
        if similarity_score is not None:
            st.write(f"**Distance from Query [Similarity Metric]**: {similarity_score:.4f}")

def search(embeddings_matrix, normalized_embeddings, query_embedding, k, distance_type):
    if distance_type == "Cosine":
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
        index.add(normalized_embeddings.astype(np.float32))
        D, I = index.search(np.array([query_embedding]), k)
        D = 1 - D
    elif distance_type == "L2":
        index = faiss.IndexFlatL2(embeddings_matrix.shape[1])
        index.add(embeddings_matrix.astype(np.float32))
        D, I = index.search(np.array([query_embedding]), k)
    elif distance_type == "Manhattan":
        D = manhattan_distance(embeddings_matrix, query_embedding)
        I = np.argsort(D)[:k].reshape(1, -1)
        D = D[I[0]]
    elif distance_type == "Chebyshev":
        D = chebyshev_distance(embeddings_matrix, query_embedding)
        I = np.argsort(D)[:k].reshape(1, -1)
        D = D[I[0]]
    return D, I

def main():
    embedding_path, metadata_path, model_type = get_cli_args()
    generate_image_embedding = get_image_embedding_function(model_type)
    embeddings_matrix, normalized_embeddings_matrix, df = load_data(embedding_path, metadata_path)

    st.title("🛋️ Furniture Similarity Search Engine")

    col1, col2 = st.columns([3, 1])
    with col1:
        query_idx_input = st.text_input('Enter Image Index for Query:', '0')
    with col2:
        k_input = st.text_input('Enter Number of Nearest Neighbors:', '4')
    k = int(k_input) + 1 if k_input.isdigit() else 5

    distance_type = st.selectbox(
        'Select Distance Measure:',
        ['L2', 'Cosine', 'Manhattan', 'Chebyshev'],
        index=0
    )

    st.subheader("🔍 Or Search by Uploading an Image")
    uploaded_file = st.file_uploader("Upload a furniture image", type=["jpg", "jpeg", "png"])

    run_search = st.button("🔍 Search Similar Items")

    if run_search:
        try:
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", width=300)

                with st.spinner("Generating embedding and description..."):
                    temp_path = "temp_uploaded_image.jpg"
                    image.save(temp_path)

                    img_emb = generate_image_embedding(temp_path)
                    desc = generate_description(temp_path)
                    text_emb = generate_text_embedding(desc)
                    query_embedding = np.hstack([img_emb, text_emb])

                    os.remove(temp_path)

                st.write("**Generated Description:**", desc)
                k -= 1
                st.divider()

            elif query_idx_input.isdigit() and 0 <= int(query_idx_input) < len(df):
                query_idx = int(query_idx_input)
                query_embedding = embeddings_matrix[query_idx]
                st.write(f"### Query Image (Index: {query_idx}) and its Metadata using {distance_type} distance")
                show_image_with_metadata(df.iloc[query_idx]['image_path'], df.iloc[query_idx].to_dict(), idx=query_idx)
                st.divider()
            else:
                st.error("Please upload an image or enter a valid image index.")
                return

            # --- Search and display results ---
            D, I = search(embeddings_matrix, normalized_embeddings_matrix, query_embedding, k, distance_type)
            if uploaded_file: 
                k += 1
            st.subheader(f"Top {k - 1} Similar Items")

            shown = 0
            for i in range(len(I[0])):
                idx = I[0][i]

                # Skip query image if applicable
                if uploaded_file is None and idx == int(query_idx_input):
                    continue

                # Avoid out-of-bounds
                if idx >= len(df):
                    continue

                try:
                    row = df.iloc[idx]
                    similarity_score = D[i] if distance_type not in ('L2', 'Cosine') else D[0][i]
                    show_image_with_metadata(row['image_path'], row.to_dict(), similarity_score=similarity_score, idx=idx)
                    st.divider()
                    shown += 1
                except Exception as e:
                    st.warning(f"Could not display result at index {idx}: {e}")

            if shown == 0:
                st.info("No similar items found to display.")

        except Exception as e:
            st.error(f"Error during search: {e}")

if __name__ == "__main__":
    main()
