import argparse
import time
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import faiss
from PIL import Image

from make_text_embeddings import generate_text_embedding
from ollama_generate_description import generate_description

st.set_page_config(page_title="Furniture Recommendation Engine", layout="wide")

def get_image_embedding_function(model_type='DINOv2'):
    if model_type.lower() == "clip":
        from make_image_embeddings import generate_image_embedding
    else:
        from make_image_embeddings_DINOv2 import generate_image_embedding
    return generate_image_embedding


class FurnitureRecommendationEngine:
    def __init__(self, model_type="DINOv2"):
        self.embeddings_matrix = None
        self.metadata_df = None
        self.image_paths = []
        self.model_type = model_type
        self.generate_image_embedding = get_image_embedding_function(model_type)

    def process_image_folder(self, folder_path):
        image_paths = list(Path(folder_path).glob('*.png')) + list(Path(folder_path).glob('*.jpg'))
        if not image_paths:
            raise FileNotFoundError(f"No images found in {folder_path}")

        image_embeddings, descriptions, desc_embeddings = [], [], []
        self.image_paths = []
        total_images = len(image_paths)

        st.info(f"Found **{total_images}** images in `{folder_path}`.")
        progress_bar = st.progress(0)
        progress_text = st.empty()

        start_time = time.time()

        for i, img_path in enumerate(image_paths):
            try:
                img_emb = self.generate_image_embedding(str(img_path))
                desc = generate_description(str(img_path))
                desc_emb = generate_text_embedding(desc)

                image_embeddings.append(img_emb)
                descriptions.append(desc)
                desc_embeddings.append(desc_emb)

                self.image_paths.append(str(img_path))
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
            finally:
                elapsed = time.time() - start_time
                items_done = i + 1
                items_left = total_images - items_done
                time_per_item = elapsed / items_done
                eta = items_left * time_per_item
                
                progress_percent = int((items_done / total_images) * 100)
                progress_bar.progress(progress_percent)

                progress_text.markdown(
                    f"**Processing {os.path.basename(img_path)}** "
                    f"({items_done}/{total_images})  \n"
                    f"Avg time/item: `{time_per_item:.2f}s`  |  ETA: `{eta:.1f}s`"
                )

        progress_text.success("Processing complete.")
        progress_bar.empty()

        # Combine embeddings
        image_embeddings = np.array(image_embeddings)
        desc_embeddings = np.array(desc_embeddings)
        self.embeddings_matrix = np.hstack([image_embeddings, desc_embeddings])

        # Create metadata dataframe
        self.metadata_df = pd.DataFrame({
            'Image Path': self.image_paths,
            'Description': descriptions
        })

        return self.embeddings_matrix, self.metadata_df

    def initialize_search_index(self):
        """Initialize FAISS index for similarity search"""
        if self.embeddings_matrix is None:
            raise ValueError("Please process images first")
            
        self.normalized_embeddings = self.embeddings_matrix / np.linalg.norm(
            self.embeddings_matrix, axis=1, keepdims=True
        )
        
        # Create indices for different distance metrics
        self.indices = {
            'L2': faiss.IndexFlatL2(self.embeddings_matrix.shape[1]),
            'Cosine': faiss.IndexFlatIP(self.embeddings_matrix.shape[1])
        }
        self.indices['L2'].add(self.embeddings_matrix.astype(np.float32))
        self.indices['Cosine'].add(self.normalized_embeddings.astype(np.float32))

    def search_similar(self, query_idx, k=5, distance_type="Cosine"):
        """Find similar items to query image"""
        if distance_type not in ['L2', 'Cosine', 'Manhattan', 'Chebyshev']:
            raise ValueError("Invalid distance type")
            
        query_embedding = self.embeddings_matrix[query_idx]
        
        if distance_type == "L2":
            D, I = self.indices['L2'].search(np.array([query_embedding]), k)
        elif distance_type == "Cosine":
            D, I = self.indices['Cosine'].search(
                np.array([query_embedding / np.linalg.norm(query_embedding)]), k
            )
            D = 1 - D  # Convert to distance
        elif distance_type == "Manhattan":
            distances = np.sum(np.abs(self.embeddings_matrix - query_embedding), axis=1)
            distances_sorted_idx = np.argsort(distances)
            D = distances[distances_sorted_idx[:k]]
            I = distances_sorted_idx[:k]
            I = I.reshape(1, -1)  
        elif distance_type == "Chebyshev":
            distances = np.max(np.abs(self.embeddings_matrix - query_embedding), axis=1)
            distances_sorted_idx = np.argsort(distances)
            D = distances[distances_sorted_idx[:k]]
            I = distances_sorted_idx[:k]
            I = I.reshape(1, -1)  
            
        return D, I

def initialize_engine():
    """Initialize the recommendation engine once"""
    if 'engine' not in st.session_state:
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_folder", type=str, required=False,
                             help="Path to folder containing furniture images")
        parser.add_argument("--index_folder", type=str, default="index",
                             help="Folder to store/load generated embeddings and dataframe")
        parser.add_argument("--type", type=str, default="DINOv2", choices=["DINOv2", "CLIP", "clip"],
                             help="Model type to use for image embeddings")
        args = parser.parse_args()

        if 'image_model_type' not in st.session_state:
            st.session_state.image_model_type = args.type

        st.session_state.engine = FurnitureRecommendationEngine(model_type=args.type)

        if args.data_folder:
            # Process images and generate embeddings
            print(f"Processing images in {args.data_folder}...")
            st.session_state.engine.process_image_folder(args.data_folder)
            st.session_state.engine.initialize_search_index()

            os.makedirs(args.index_folder, exist_ok=True)
            folder_name = Path(args.data_folder).stem
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_dir = Path(args.index_folder) / f"{folder_name}_{timestamp}"
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save generated data
            np.save(save_dir / 'generated_embeddings.npy', st.session_state.engine.embeddings_matrix)
            st.session_state.engine.metadata_df.to_csv(save_dir / 'generated_dataframe.csv', index=False)
            print(f"Saved embeddings and dataframe in {save_dir}")
        else:
            # Load latest available saved data
            try:
                index_base = Path(args.index_folder)
                if not index_base.exists():
                    raise FileNotFoundError("No index folder found")
                
                # Get latest subfolder by creation time
                subfolders = [f for f in index_base.iterdir() if f.is_dir()]
                latest_folder = max(subfolders, key=os.path.getmtime)

                print(f"Loading from latest index folder: {latest_folder}")
                st.session_state.engine.embeddings_matrix = np.load(latest_folder / 'generated_embeddings.npy')
                st.session_state.engine.metadata_df = pd.read_csv(latest_folder / 'generated_dataframe.csv')
                st.session_state.engine.image_paths = st.session_state.engine.metadata_df['Image Path'].tolist()
                st.session_state.engine.initialize_search_index()
                print(f"Loaded embeddings and metadata from {latest_folder}")
            except Exception as e:
                print(f"Error loading pre-generated data: {e}")
                st.error("Failed to load index. Provide --data_folder to generate new embeddings.")
                return False
    return True

def main():
    if not initialize_engine():
        return

    st.title("🛋️ Furniture Recommendation Engine")
    st.markdown(
        "Welcome to the **Furniture Similarity Search Tool**. "
        "This tool allows you to find similar furniture items from a dataset based on their features and visual appearance."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        query_idx_input = st.text_input('Enter Image Index for Query:', '0')
    with col2:
        k_input = st.text_input('Enter Number of Nearest Neighbors:', '4')
    k = int(k_input) + 1 if k_input.isdigit() else 4

    distance_type = st.selectbox(
        'Select Distance Measure:',
        ['L2', 'Cosine', 'Manhattan', 'Chebyshev'],
        index=0
    )

    st.subheader("🔍 Search by Uploading an Image")
    uploaded_file = st.file_uploader("Upload a furniture image", type=["jpg", "jpeg", "png"])

    run_search = st.button("🔍 Search Similar Items")

    if run_search:
        try:
            # Case 1: If image uploaded
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", width=300)

                with st.spinner("Generating embedding and description..."):
                    temp_path = "temp_uploaded_image.jpg"
                    image.save(temp_path)

                    generate_image_embedding = get_image_embedding_function(model_type=st.session_state.image_model_type)
                    uploaded_img_emb = generate_image_embedding(image_path=temp_path)
                    uploaded_desc = generate_description(image_path=temp_path)
                    uploaded_text_emb = generate_text_embedding(uploaded_desc)
                    uploaded_combined_emb = np.hstack([uploaded_img_emb, uploaded_text_emb])

                    if distance_type == "Cosine":
                        uploaded_combined_emb = uploaded_combined_emb / np.linalg.norm(uploaded_combined_emb)

                    # Perform search
                    if distance_type in ["L2", "Cosine"]:
                        index = st.session_state.engine.indices[distance_type]
                        D, I = index.search(np.array([uploaded_combined_emb]).astype(np.float32), k-1)
                        D = 1 - D if distance_type == "Cosine" else D
                    elif distance_type == "Manhattan":
                        distances = np.sum(np.abs(st.session_state.engine.embeddings_matrix - uploaded_combined_emb), axis=1)
                        distances_sorted_idx = np.argsort(distances)
                        D = distances[distances_sorted_idx[:k-1]]
                        I = distances_sorted_idx[:k-1].reshape(1, -1)
                    elif distance_type == "Chebyshev":
                        distances = np.max(np.abs(st.session_state.engine.embeddings_matrix - uploaded_combined_emb), axis=1)
                        distances_sorted_idx = np.argsort(distances)
                        D = distances[distances_sorted_idx[:k-1]]
                        I = distances_sorted_idx[:k-1].reshape(1, -1)

                    st.write("**Generated Description:**", uploaded_desc)
                    st.divider()

            # Case 2: Use query index if valid and no file uploaded
            elif query_idx_input.isdigit() and 0 <= int(query_idx_input) < len(st.session_state.engine.metadata_df):
                query_idx = int(query_idx_input)
                D, I = st.session_state.engine.search_similar(query_idx, k, distance_type)

                query_img = Image.open(st.session_state.engine.image_paths[query_idx])
                st.image(query_img, caption=f"Query Image (Index: {query_idx})", width=300)
                st.write("**Description:**", st.session_state.engine.metadata_df.iloc[query_idx]['Description'])
                st.divider()
            else:
                st.error("Please upload an image or enter a valid image index.")
                return

            # Show results
            st.subheader(f"Top {k - 1} Similar Items ({distance_type} distance)")
            dist_list = D[0] if distance_type in ("L2", "Cosine") else D
            for i, (idx, dist) in enumerate(zip(I[0], dist_list)):
                if uploaded_file is None and idx == int(query_idx_input):
                    continue
                col1, col2 = st.columns([1, 3])
                with col1:
                    sim_img = Image.open(st.session_state.engine.image_paths[idx])
                    st.image(sim_img, caption=f"Image {idx}", use_column_width=True)
                with col2:
                    st.write(st.session_state.engine.metadata_df.iloc[idx]['Description'])
                    st.write(f"**Distance from Query [Similarity Metric]**: {dist:.3f}")
                st.divider()

        except Exception as e:
            st.error(f"Error during search: {str(e)}")
        finally:
            if os.path.exists("temp_uploaded_image.jpg"):
                os.remove("temp_uploaded_image.jpg")

if __name__ == "__main__":
    main()