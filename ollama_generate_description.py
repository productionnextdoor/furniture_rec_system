import base64
import pandas as pd
import ollama


##########################################################################################################
######## This file was used to generate the whole Dataset csv file (with description) ####################
##########################################################################################################


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

def generate_description(image_path):
    base64_image=encode_image_to_base64(image_path)
    response = ollama.chat(
        model='llama3.2-vision:11b-instruct-q8_0',
        messages=[{
            'role': 'user',
            'content': """Generate me a short and 'to the point' description of the furniture piece shown in the image capturing all the salient 
                          and nuanced features. Including color, number of legs, print etc; where ever it is applicable.
                          Do not comment about the image properties (e.g. furniture's orientation in the image.)
                          The output should be in one paraghraph. Maintain uniformity in all yout outputs. Keep the output to a length of 
                          60-75 words.
                          """,
            'images': [base64_image]
        }]
    )
    return response['message']['content']
    
def generate_descriptions_for_dataset(df):
    descriptions = []

    for idx, row in df.iterrows():  
        print(f"Generateing description for row no. {idx+1}")
        image_path = row['Image Path']
        description = generate_description(image_path)
        descriptions.append(description)

        # Print the generated description for debugging
        # print(f"Generated Description for image {image_path}: {description}")

    df['Generated Description'] = descriptions

    return df


# df = pd.read_csv('dataset/full_furniture_dataset.csv')  


# df_with_descriptions = generate_descriptions_for_dataset(df)

# print(df_with_descriptions[['Image Path', 'Generated Description']])

# df_with_descriptions.to_csv('dataset/full_furniture_dataset_with_descriptions.csv', index=False)