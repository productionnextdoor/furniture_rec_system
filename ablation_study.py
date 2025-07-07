import numpy as np
import pandas as pd
import faiss
import plotly.express as px


#####################################################################################################
######## This file was used to perform the Ablation Study to see the effect of each component #######
######## The Graphs can be seen in the graphs/ablation folder #######################################
#####################################################################################################


def load_embeddings():
    image_embeddings_clip = np.load('dataset/clip_embeddings.npy')
    image_embeddings_dino = np.load('dataset/image_embeddings_DINOv2_combined.npy')
    metadata_embeddings = np.load('dataset/metadata_embeddings.npy')
    text_embeddings = np.load('dataset/text_embeddings.npy')

    return {
        "Image_clip": image_embeddings_clip,
        "Image_dino": image_embeddings_dino,
        "Metadata": metadata_embeddings,
        "Text": text_embeddings
    }


def normalize(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

def create_ablation_embeddings(embeddings_dict):
    return {
        "Baseline_clip (Approach 1: Image_clip + Metadata)": np.hstack([embeddings_dict["Image_clip"], embeddings_dict["Metadata"]]),
        "Baseline_dino (Approach 1: Image_dino + Metadata)": np.hstack([embeddings_dict["Image_dino"], embeddings_dict["Metadata"]]),
        "CLIP Image": embeddings_dict["Image_clip"],
        "DINO Image": embeddings_dict["Image_dino"], 
        "No Metadata (Image_clip + Text)": np.hstack([embeddings_dict["Image_clip"], embeddings_dict["Text"]]),
        "No Metadata (Image_dino + Text)": np.hstack([embeddings_dict["Image_dino"], embeddings_dict["Text"]]),
        "Approach 2 (Image_clip + Metadata + Text)": np.hstack([embeddings_dict["Image_clip"], embeddings_dict["Metadata"], embeddings_dict["Text"]]),
        "Approach 2 (Image_dino + Metadata + Text)": np.hstack([embeddings_dict["Image_dino"], embeddings_dict["Metadata"], embeddings_dict["Text"]])
    }

def perform_similarity_search(index, embeddings, k):
    distances, _ = index.search(embeddings, k)
    distances = (distances - distances.min()) / (distances.max() - distances.min())
    similarity_scores = 1 - distances
    return similarity_scores.mean(axis=1)

def ablation_study(index, df, embeddings, category_column, k, approach_name):
    df_copy = df.copy()
    similarity_scores = perform_similarity_search(index, embeddings, k)
    df_copy["Mean Similarity"] = similarity_scores
    category_sim = df_copy.groupby(category_column).agg(
        Mean_Similarity=("Mean Similarity", "mean"),
        Variance_Similarity=("Mean Similarity", "var"),
        Median_Similarity=("Mean Similarity", "median")
    ).reset_index()
    category_sim["Approach"] = approach_name
    df_copy["Approach"] = approach_name
    return category_sim, df_copy

def plot_ablation_results(category_results, category_column, selected_approaches):
    category_results_filtered = category_results[category_results["Approach"].isin(selected_approaches)]
    metrics = ["Mean_Similarity", "Variance_Similarity", "Median_Similarity"]
    titles = {
        "Mean_Similarity": "Mean Similarity by Category",
        "Variance_Similarity": "Variance in Similarity Scores",
        "Median_Similarity": "Median Similarity Scores by Category"
    }

    figures = []
    for metric in metrics:
        fig = px.bar(category_results_filtered, 
                     x=category_column, 
                     y=metric, 
                     color="Approach",
                     barmode="group", 
                     title=titles[metric],
                     labels={metric: metric.replace("_", " ")})
        figures.append(fig)
    return figures


def plot_abalation_boxplot(df, category_column, selected_approaches):
    df_filtered = df[df["Approach"].isin(selected_approaches)]
    fig = px.box(df_filtered, 
                 x=category_column, 
                 y="Mean Similarity", 
                 color="Approach", 
                 title="Ablation Study: Similarity Score Distribution by Category",
                 points="outliers",  
                 boxmode="group")
    return fig


embeddings_dict = load_embeddings()
embeddings_dict = create_ablation_embeddings(embeddings_dict)

indices = {}
for name, embeddings in embeddings_dict.items():
    normalized_embeddings = normalize(embeddings)
    index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
    index.add(normalized_embeddings)
    indices[name] = index

df = pd.read_csv('dataset/full_furniture_dataset_with_descriptions.csv')

category_column = "Category"
top_k = 10
category_results = []
df_list = []

for approach_name in indices.keys():
    index = indices[approach_name]
    embeddings = embeddings_dict[approach_name]
    results, df_copy = ablation_study(index, df, embeddings, category_column, top_k, approach_name)
    print("Got results for the study.")
    category_results.append(results)
    df_list.append(df_copy)

category_results = pd.concat(category_results)
df_copy = pd.concat(df_list)
df_copy["Normalized Similarity"] = df_copy.groupby(category_column)["Mean Similarity"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
)

selected_approaches = list(indices.keys())

bar_figures = plot_ablation_results(category_results, category_column, selected_approaches)
for i, fig in enumerate(bar_figures):
    fig.write_html(f"ablation_study_bar_{i}.html")

boxplot_figure = plot_abalation_boxplot(df_copy, category_column, selected_approaches)
boxplot_figure.write_html("ablation_study_boxplot.html")

print("All graphics generated and saved as HTML files.")
