import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import faiss
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from ast import literal_eval


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['Category'] != 'beds'].reset_index(drop=True)
    df["embedding"] = df["embedding"].apply(literal_eval)  
    embeddings = np.vstack(df["embedding"].values)
    return df, embeddings

df_1, embeddings_1 = load_data('clip_metadata_embed_dedup.csv')
df_2, embeddings_2 = load_data('clip_metadata_desc_embed_dedup.csv')

def normalize(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

embeddings_1 = embeddings_1.astype('float32')
embeddings_2 = embeddings_2.astype('float32')
faiss.normalize_L2(embeddings_1)
faiss.normalize_L2(embeddings_2)


index_1 = faiss.IndexFlatIP(embeddings_1.shape[1])
index_2 = faiss.IndexFlatIP(embeddings_2.shape[1])

index_1.add(embeddings_1)
index_2.add(embeddings_2)

top_k = 10  
category_column = "Category"


def get_similar_items(index, embeddings, df, top_k):
    similarities, neighbors = index.search(embeddings, top_k)
    results = []
    for i, (sim, idxs) in enumerate(zip(similarities, neighbors)):
        items = df.iloc[idxs].copy()
        items["Similarity Score"] = sim
        results.append(items)
    return results

results_1 = get_similar_items(index_1, embeddings_1, df_1, top_k)
results_2 = get_similar_items(index_2, embeddings_2, df_2, top_k)

def category_consistency(results, df, category_column):
    consistencies = []
    for i, items in enumerate(results):
        query_category = df.iloc[i][category_column]
        same_category_count = sum(items[category_column] == query_category)
        consistencies.append(same_category_count / len(items))
    return np.mean(consistencies)

consistency_1 = category_consistency(results_1, df_1, category_column)
consistency_2 = category_consistency(results_2, df_2, category_column)

with open("evaluation_corrected.txt", "w") as f:
    f.write(f"Category Consistency for \n \t Approach 1: {consistency_1:.4f} \n \t Approach 2: {consistency_2:.4f}\n\n")


def plot_tsne(embeddings_1, embeddings_2, categories):
    tsne_1 = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings_1)
    tsne_2 = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings_2)
    
    df_tsne_1 = pd.DataFrame(tsne_1, columns=["x", "y"])
    df_tsne_1["Category"] = categories
    df_tsne_1["Approach"] = "Approach 1"
    
    df_tsne_2 = pd.DataFrame(tsne_2, columns=["x", "y"])
    df_tsne_2["Category"] = categories
    df_tsne_2["Approach"] = "Approach 2"
    
    df_tsne = pd.concat([df_tsne_1, df_tsne_2])
    
    fig = px.scatter(df_tsne, x="x", y="y", color="Category", facet_col="Approach",
                     title="t-SNE Visualization of Embeddings",
                     opacity=0.7)
    fig.write_html("tsne_plot_wo_beds_corrected.html")

plot_tsne(embeddings_1, embeddings_2, df_1[category_column])

def plot_similarity_histogram(index_1, index_2, embeddings_1, embeddings_2, k):
    scores_1, _ = index_1.search(embeddings_1, k)
    scores_2, _ = index_2.search(embeddings_2, k)

    df_scores = pd.DataFrame({
        "Similarity Score": np.concatenate([scores_1.flatten(), scores_2.flatten()]),
        "Approach": ["Approach 1"] * len(scores_1.flatten()) + ["Approach 2"] * len(scores_2.flatten())
    })

    plt.figure(figsize=(10, 6))
    sns.histplot(df_scores, x="Similarity Score", hue="Approach", bins=50, kde=True, alpha=0.6)

    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    plt.title("Histogram of Similarity Scores")

    plt.savefig("similarity_histogram_wo_beds_corrected.png", dpi=300)
    plt.show()

plot_similarity_histogram(index_1, index_2, embeddings_1, embeddings_2, top_k)

def kl_divergence(index_1, index_2, embeddings_1, embeddings_2, k):
    scores_1, _ = index_1.search(embeddings_1, k)
    scores_2, _ = index_2.search(embeddings_2, k)

    p = np.histogram(scores_1.flatten(), bins=100, density=True)[0]
    q = np.histogram(scores_2.flatten(), bins=100, density=True)[0]
    
    p += 1e-9
    q += 1e-9
    kl_div = entropy(p, q)
    with open("evaluation_corrected.txt", "a") as f:
        f.write(f"KL Divergence between Approach 1 and 2: {kl_div:.4f}\n\n")

kl_divergence(index_1, index_2, embeddings_1, embeddings_2, top_k)


def similarity_statistics(index_1, index_2, embeddings_1, embeddings_2, k):
    scores_1, _ = index_1.search(embeddings_1, k)
    scores_2, _ = index_2.search(embeddings_2, k)

    mean_1, var_1 = np.mean(scores_1), np.var(scores_1)
    mean_2, var_2 = np.mean(scores_2), np.var(scores_2)

    with open("evaluation_corrected.txt", "a") as f:
        f.write(f"""Similarity Statistics - \n \t Approach 1 - Mean Similarity: {mean_1:.4f}, Variance: {var_1:.4f} \n \t
                Approach 2 - Mean Similarity: {mean_2:.4f}, Variance: {var_2:.4f}\n\n""")
    
similarity_statistics(index_1, index_2, embeddings_1, embeddings_2, top_k)


def plot_similarity_by_category(index_1, index_2, df_1, df_2, embeddings_1, embeddings_2, category_column, k):
    scores_1, _ = index_1.search(embeddings_1, k)
    scores_2, _ = index_2.search(embeddings_2, k)

    df_1["Mean Similarity"] = scores_1.mean(axis=1)
    df_2["Mean Similarity"] = scores_2.mean(axis=1)

    df_combined = pd.concat([
        df_1[[category_column, "Mean Similarity"]].assign(Approach="Approach 1"),
        df_2[[category_column, "Mean Similarity"]].assign(Approach="Approach 2")
    ])

    fig = px.box(df_combined, x=category_column, y="Mean Similarity", color="Approach",
             title="Similarity Score Distribution by Category", points="outliers", 
             boxmode="group")
    fig.write_html("similarity_by_category_wo_beds_corrected.html")

plot_similarity_by_category(index_1, index_2, df_1, df_2, embeddings_1, embeddings_2, category_column, top_k)


def high_quality_match_percentage(index_1, index_2, embeddings_1, embeddings_2, threshold, k):
    scores_1, _ = index_1.search(embeddings_1, k)
    scores_2, _ = index_2.search(embeddings_2, k)

    high_match_1 = (scores_1 > threshold).sum() / scores_1.size * 100
    high_match_2 = (scores_2 > threshold).sum() / scores_2.size * 100

    with open("evaluation_corrected.txt", "a") as f:
        f.write(f""" High Quality Matches: \n \t Approach 1 - (> {threshold}): {high_match_1:.2f}% \n \t
                Approach 2 - (> {threshold}): {high_match_2:.2f}% \n\n""")
    
for threshold in [0.2, 0.4, 0.6, 0.8]:
    high_quality_match_percentage(index_1, index_2, embeddings_1, embeddings_2, threshold=threshold, k=top_k)

def nearest_neighbor_consistency(index_1, index_2, embeddings_1, embeddings_2, k):
    _, indices_1, = index_1.search(embeddings_1, k)
    _, indices_2, = index_2.search(embeddings_2, k)

    jaccard_similarities = [
        len(set(indices_1[i]) & set(indices_2[i])) / len(set(indices_1[i]) | set(indices_2[i]))
        for i in range(len(indices_1))
    ]

    with open("evaluation_corrected.txt", "a") as f:
        f.write(f"Average Nearest Neighbor Consistency (Jaccard Similarity): {np.mean(jaccard_similarities):.4f}\n\n")

nearest_neighbor_consistency(index_1, index_2, embeddings_1, embeddings_2, top_k)


def plot_category_wise_similarity(index_1, index_2, df_1, df_2, embeddings_1, embeddings_2, category_column, k):
    scores_1, _ = index_1.search(embeddings_1, k)
    scores_2, _ = index_2.search(embeddings_2, k)

    df_1["Mean Similarity"] = scores_1.mean(axis=1)
    df_2["Mean Similarity"] = scores_2.mean(axis=1)

    category_sim_1 = df_1.groupby(category_column)["Mean Similarity"].mean().reset_index()
    category_sim_2 = df_2.groupby(category_column)["Mean Similarity"].mean().reset_index()

    category_sim_1["Approach"] = "Approach 1"
    category_sim_2["Approach"] = "Approach 2"
    
    combined_df = pd.concat([category_sim_1, category_sim_2])

    fig = px.bar(combined_df, x=category_column, y="Mean Similarity", color="Approach", barmode="group",
                 title="Category-wise Similarity Comparison", labels={"Mean Similarity": "Average Similarity Score"})
    
    fig.write_html("category_wise_similarity_wo_beds_corrected.html")

plot_category_wise_similarity(index_1, index_2, df_1, df_2, embeddings_1, embeddings_2, category_column, top_k)


def intraclass_distance(df, embeddings, category_column):
    unique_categories = df[category_column].unique()
    distances = []
    for category in unique_categories:
        indices = df[df[category_column] == category].index
        if len(indices) > 1:
            subset_embeddings = embeddings[indices]
            sim_matrix = cosine_similarity(subset_embeddings)
            avg_dist = np.mean(sim_matrix)
            distances.append(avg_dist)
    return np.mean(distances)

intraclass_1 = intraclass_distance(df_1, embeddings_1, category_column)
intraclass_2 = intraclass_distance(df_2, embeddings_2, category_column)

with open('evaluation_corrected.txt', 'a') as f:
    f.write(f"""Intraclass Distance: \n \t Approach 1: {intraclass_1:.4f} \n Approach2: {intraclass_2:.4f}\n\n""")