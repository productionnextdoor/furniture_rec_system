import numpy as np
import pandas as pd


###########################################################################################
######## This file was used to generate the Datasets to comapre both the approaches #######
###########################################################################################


def combine_embeddings(image_embedding, text_embedding, description_embedding=None):
    if not description_embedding is None:
        combined_embedding = np.hstack([image_embedding, text_embedding, description_embedding])
    else:
        combined_embedding = np.hstack((image_embedding, text_embedding))
    return combined_embedding


image_embeddings = np.load("dataset/clip_embeddings.npy")
metadata_embeddings = np.load("dataset/metadata_embeddings.npy")
description_embeddings = np.load('dataset/text_embeddings.npy')


df_desc = pd.read_csv('dataset/full_furniture_dataset_with_descriptions.csv')  
df_desc = df_desc[df_desc['Category'] != 'beds'].reset_index(drop=True)
df_wo_desc = df_desc.drop(['Generated Description'], axis = 1)


clip_metadata = np.array(combine_embeddings(image_embeddings, metadata_embeddings), dtype=np.float32)
clip_metadata_desc = np.array(combine_embeddings(image_embeddings, metadata_embeddings + description_embeddings), dtype=np.float32)

np.save('dataset/embeddings.npy', clip_metadata_desc)

df_wo_desc['embedding'] = clip_metadata.tolist()
df_desc['embedding'] = clip_metadata_desc.tolist()

df_wo_desc.to_csv('dataset/clip_metadata_embed.csv', index=False)
df_desc.to_csv('dataset/clip_metadata_desc_embed.csv', index=False)