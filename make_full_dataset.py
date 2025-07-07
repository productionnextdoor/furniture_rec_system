import os
import io
import re
import pandas as pd
from PIL import Image

image_dir = 'dataset/dataset22/Bonn_Furniture_Styles_Dataset/'
metadata_dir = 'dataset/dataset22/Bonn_Furniture_Styles_Dataset/metadata'
dataset = []

##########################################################################
######## This file was used to generate the whole Dataset csv file #######
##########################################################################


def get_dimensions_and_weights(dim_weight_metadata):
    # Initialize variables to store the results
    dimensions = [None, None, None]  # [width, depth, height]
    weight = None
    
    # Regex patterns to extract the values
    dimension_pattern = re.compile(r'(\w)\s*(\d+(\.\d+)?)')  # Looks for W <value>, D <value>, H <value>
    weight_pattern = re.compile(r'(\d+(\.\d+)?)\s*lb\.')  # Looks for lb <value>
    
    # Match dimensions
    dimension_matches = dimension_pattern.findall(dim_weight_metadata)
    for match in dimension_matches:
        if match[0] == 'W':
            dimensions[0] = float(match[1])  # width
        elif match[0] == 'D':
            dimensions[1] = float(match[1])  # depth
        elif match[0] == 'H':
            dimensions[2] = float(match[1])  # height
    
    # Match weight
    weight_match = weight_pattern.search(dim_weight_metadata)
    if weight_match:
        weight = float(weight_match.group(1))
    
    return dimensions, weight


def parse_metadata(metadata, category):
    metadata_parts = metadata.split(';')
    if len(metadata_parts) == 2:
        metadata_dict = {
            'Brand': None,
            'Dimensions [W X D X H] (in inches)': None,
            'Weight (in lb.)': None,
            'Color': None,
            'Material': None,
            'Category': category,
            'Style': metadata_parts[1].strip()
        }
    else:
        dimensions, weight = get_dimensions_and_weights(metadata_parts[1])
        metadata_dict = {
        'Brand': metadata_parts[0].split(':')[-1],
        'Dimensions [W X D X H] (in inches)': dimensions,
        'Weight (in lb.)': weight,
        'Color': metadata_parts[2].strip() if len(metadata_parts) > 2 else None,
        'Material': metadata_parts[3].strip() if len(metadata_parts) > 3 else None,
        'Category': category,
        'Style': metadata_parts[5].strip() if len(metadata_parts) > 5 else None
    }
    return metadata_dict

def process_files(file_name):
    with open(os.path.join(metadata_dir, file_name), 'r') as f:
        lines = f.readlines()

    print(f"Working in file: {file_name}")

    for line in lines:
        parts = line.split('\t')

        image_path = parts[1].strip()
        metadata = parts[-1].strip()

        image_full_path = os.path.join(image_dir, image_path)

        if os.path.exists(image_full_path):
            with open(image_full_path, 'rb') as img_file:
                img_data = img_file.read()
            img = Image.open(io.BytesIO(img_data))

            metadata_dict = parse_metadata(metadata, category=file_name[:-4])

            dataset.append({
                'Image Path': image_full_path,
                'Image': img,
                **metadata_dict
            })


for file in os.listdir(metadata_dir):
    process_files(file)

df = pd.DataFrame(dataset)

df.drop(columns=['Image'], inplace=True)  
df.to_csv('dataset/full_furniture_dataset.csv', index=False)


print("Dataset created successfully!")
