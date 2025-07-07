import os
import base64
import requests
from dotenv import load_dotenv
import pandas as pd


############################################################################################
######## This file was used to generate sampled Dataset (with descirption) #################
######## Containing 25 items for each Category #############################################
############################################################################################


load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

def generate_description(image_path, metadata, exemplary_description):
    base64_image=encode_image_to_base64(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": 
                f"Here is an example description: '{exemplary_description}'.\n" \
                f"Now, based on the image and the following attributes: {metadata}, generate a short description for the image."
            }, {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }]
        }],
        "max_tokens": 1100
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=None).json()
    return response['choices'][0]['message']['content']

    
def generate_descriptions_for_dataset(df, exemplary_description=None):
    descriptions = []

    for idx, row in df.iterrows():
        print(f"Generating description for row no. {idx+1}")
        image_path = row['Image Path']
        metadata = {col: row[col] for col in df.columns if col not in ['Image Path']}

        description = generate_description(image_path, metadata, exemplary_description)
        descriptions.append(description)

        # Print the generated description for debugging
        # print(f"Generated Description for image {image_path}: {description}")

    df['Generated Description'] = descriptions

    return df

exemplary_description = "Artiko stellt ein innovatives Möbeldesign dar, das als multifunktionale Lösung für das Zuhause konzipiert ist. Es kombiniert die Funktionalität eines Sideboards und eines Schreibtisches mit einem integrierten Hocker und bietet somit zwei unterschiedliche Modi für Arbeit und Wohnen. Im Workmodus kann die mittlere Schublade herausgezogen und als bequemer Hocker positioniert werden, um den Schreibtischbereich zu aktivieren. Für den Homemodus wird die Schublade einfach zurückgeschoben, wodurch das Artiko wieder in ein elegantes Sideboard für das Wohnzimmer transformiert wird. Dieses Möbelstück besticht nicht nur durch seine regionale und dauerhafte Materialwahl, sondern auch durch seine einfache und zeitsparende Umbaufähigkeit, die den Wechsel zwischen den Modi mühelos gestaltet."


df = pd.read_csv('dataset/furniture_dataset.csv')  


df_with_descriptions = generate_descriptions_for_dataset(df, exemplary_description)

# print(df_with_descriptions[['Image Path', 'Generated Description']])

df_with_descriptions.to_csv('furniture_dataset_with_descriptions.csv', index=False)