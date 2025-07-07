
# Image-Based Furniture Recommendation System

This project implements a visual recommendation system for furniture using image embeddings, metadata, and generated text descriptions. It includes tools for evaluating different embedding strategies and analyzing model behavior to detect overfitting.


## Overview

The goal is to recommend visually and semantically similar furniture items by leveraging:
- **Visual Features (Image Embeddings)**
- **Product Metadata**
- **Generated Text Descriptions**

We explore multiple approaches and perform an ablation study comparing the effectiveness of CLIP and DINOv2 embeddings. The system also includes tools to detect overfitting in learned representations.


## Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```


## Repository Structure

```
.
├── dataset/
├── graphs/
│   ├── ablation/
│   ├── evaluation/
│   └── overfitting/
├── ablation_study.py
├── clip_overfitting.py
├── combine_dino_embeddings.py
├── create_datasets.py
├── evaluation.py
├── evaluation.txt
├── generate_description.py
├── make_ablation_image_embeddings_DINOv2.py
├── make_embeddings.py
├── make_full_dataset.py
├── make_image_embeddings.py
├── make_image_embeddings_DINOv2.py
├── make_sampled_dataset.py
├── make_text_embeddings_bge.py
├── make_text_embeddings.py
├── ollama_generate_description.py
├── overfitting.txt
├── recommendation_app.py
├── readme.md
```

> **Note**: The dataset folder is not present in the repository due to size constraints as well as proprietary issues.


## Usage 


### `image_processor.py` - Unified Image Processing Pipeline

**Purpose**:  
Provides a single interface for converting raw furniture images into:
1. **Visual embeddings** (DINOv2 or CLIP)
2. **LLM-generated descriptions** (llama3.2-vision) *
3. **Text embeddings** (MiniLM)

To Setup the Vision Language Model (VLM) - 
1. Setup Ollama - Using the instructions given [here](https://ollama.com/). 
2. Once the download is finished. Pull model using command: `ollama pull llama3.2-vision:11b-instruct-q8_0`
3. To run this model efficiently a VRAM of 13 GB or more is required. Otherwise the output generation will be very slow.
4. This model is chosen after considering the size and resource requirements. For more information please check [here](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/).

**Usage Examples**:

1. **Programmatic**:
```python
from image_processor import process_image

img_emb, description, desc_emb = process_image("path/to/image.png", model_type="CLIP")
```

2. **Command Line**:
```bash
# Using default DINOv2
python image_processor.py --image_path dataset/sample_chair.png

# Using CLIP
python image_processor.py --image_path dataset/sample_chair.png --type CLIP
```


### `batch_image_processor.py`

This script processes a large collection of images by generating:

* Image embeddings (via `generate_image_embedding`) using either `DINOv2` (default) or `CLIP`
* AI-generated textual descriptions (via `generate_description`)
* Embeddings of the descriptions (via `generate_text_embedding`)

It processes the images in **batches of 1000**, saves intermediate batch files, and finally merges them into a single `.npy` file and a `.csv` file. It supports recursive directory traversal and shows a progress bar during processing.

#### Features

* Batch-wise processing for memory efficiency
* Supports nested directories (recursive image loading)
* Supports two vision models: `DINOv2` (default) and `CLIP` (via `--type` argument)
* Saves intermediate results in a timestamped folder under `dataset/`
* Final `.npy` file contains concatenated image and text embeddings
* Final `.csv` file contains image paths and generated descriptions
* Deletes intermediate batch files after final merge
* Shows a real-time progress bar using `tqdm`

#### Usage

```bash
# Using DINOv2 (default)
python batch_image_processor.py --image_dir path/to/images

# Using CLIP instead of DINOv2
python batch_image_processor.py --image_dir path/to/images --type CLIP
```

* All image formats `.jpg`, `.jpeg`, `.png`, `.bmp` are supported.
* The script will recursively search through subdirectories.

#### Output

Outputs are saved under:
`dataset/<image_folder_name>_<timestamp>/`

Example:

```
dataset/my_images_20250506_141520/
├── final_embeddings.npy     # Combined image + description embeddings
├── final_data.csv           # Image paths + generated descriptions
```

Intermediate `batch_*.npy` and `batch_*.csv` files are automatically deleted after finalization.


### Recommendation Application (`recommendation_app.py`) - Streamlit-based Furniture Similarity Search

**Key Features**:
- Load large .npy and .csv datasets directly via CLI args (no UI file uploads needed)
- Perform similarity search using given embeddings or by uploading a new image
- Interactive interface for querying similar furniture items
- Supports 4 distance metrics with visual comparisons
- Displays images with description and similarity scores
- Supports dynamic image embedding model selection (`DINOv2` or `CLIP`) via `--type` argument

**Distance Metrics Implementation**

| Metric | Formula | Use Case | Pros | Cons |
|--------|---------|----------|------|------|
| **L2 (Euclidean)** | `√Σ(a_i - b_i)²` | General-purpose similarity | Intuitive, works well for balanced features | Sensitive to feature scales |
| **Cosine** | `1 - (A·B)/(‖A‖‖B‖)` | Directional similarity (ignores magnitude) | Ideal for text/CLIP embeddings | Loses distance information |
| **Manhattan (L1)** | `Σ\|a_i - b_i\|` |  Robust to outliers | Handles sparse data well | Less sensitive to small variations |
| **Chebyshev (L∞)** | `max(\|a_i - b_i\|)` | Worst-axis similarity | Identifies maximum feature difference | Ignores other dimensions |

**Why Multiple Metrics?**
1. **Comparative Analysis**  
   - *Cosine* best captures semantic similarity for embeddings  
   - *L2* provides geometric intuition in feature space  
   - *L1/L∞* test robustness to noise and outliers  

2. **Domain Adaptation**  
   - Image embeddings respond differently to each metric  
   - Helps identify optimal similarity measure 

3. **Failure Mode Detection**  
   - Discrepancies between metrics reveal edge cases  
   - Example: High cosine but low L2 similarity indicates magnitude mismatches  

**App Components**
```python
Core Functions:
- get_cli_args() and load_data(): Accepts and loads data
- get_image_embedding_function(): Dynamically selects DINOv2 or CLIP
- show_image_with_metadata(): Renders results in Streamlit columns  
- search(): Handles search logic for all metrics  
- Distance implementations: FAISS (L2/Cosine), NumPy (L1/L∞)  
```

**Usage**:
```bash
# Default (DINOv2)
streamlit run recommendation_app.py -- --embedding_path path/to/embeddings.npy --dataframe_path path/to/dataframe.csv

# Using CLIP
streamlit run recommendation_app.py -- --embedding_path path/to/embeddings.npy --dataframe_path path/to/dataframe.csv --type CLIP
```

**Output**:
- Interactive web interface at `localhost:8501`
- Displays:
  - Query image + description 
  - Top-k similar items
  - Visualized distance scores


### `recommendation_engine.py` - Unified Recommendation Pipeline

#### Features

* **Image Embeddings** via pretrained vision models (e.g., DINOv2 or CLIP).
* **Auto Descriptions** generated using an AI model (e.g., LLaMA3.2 via Ollama).
* **Text Embeddings** for semantic context (e.g., MiniLM).
* **Multimodal Search** combining visual and textual features.
* **Distance Metrics**: Cosine, L2, Manhattan, Chebyshev.
* **Indexed Results Storage** in timestamped folders under `/index`.
* **Streamlit UI** for interactive querying and visualization.
* **Search by Uploading an Image** — upload a new image and find its closest matches from the dataset.

#### Usage

##### To generate embeddings and start the app:

```bash
# Default (DINOv2)
streamlit run recommendation_engine.py -- --data_folder /path/to/your/images

# Using CLIP
streamlit run recommendation_engine.py -- --data_folder /path/to/your/images --type CLIP
```

This processes the images and saves:

* `generated_embeddings.npy`
* `generated_dataframe.csv`

To a subfolder under `index/` named `<folder_name>_<timestamp>`


##### To load the latest saved index:

```bash
streamlit run recommendation_engine.py
```

#### Streamlit UI Overview

* **Image Index Search**: Choose a query image by index.
* **Image Upload Search**: Upload any furniture image and search for similar items in the dataset.
* **Number of Neighbors**: Choose how many similar items to retrieve.
* **Distance Metric**: Choose how similarity is computed (Cosine, L2, Manhattan, Chebyshev).
* **Results**: View query/uploaded image, generated description, and top similar items with distances.

#### Output Example

```
index/
└── chairs_20250504_144501/
    ├── generated_embeddings.npy
    └── generated_dataframe.csv
```

#### Notes

* Run with `streamlit run`, **not** `python`, to avoid session state issues.
* If no `--data_folder` is provided, the app loads the latest saved index from `index/`.

#### System Usage Recommendation

> **Note:** This application has been tested on a system with the following specifications:
>
> * **Disk Space:** 500 GB total
> * **RAM:** 16 GB
>
> For reliable performance and to avoid memory overload or excessive processing time:
>
> * **Do not use folders containing more than 1,000 images**, or
> * **Do not exceed a total image size of 100 MB per dataset folder**
>
> Processing large image datasets may lead to high memory usage, slow performance, or crashes, especially during embedding generation and search index initialization.



## Overall Workflow

Firstly, the two approaches are evaluated to embed the furniture data (images + metadata) using the **Bonn Furniture Dataset**. The goal is to compare the performance of embeddings with and without generated image descriptions.

### Dataset
The **Bonn Furniture Dataset** ([Paper](https://cg.cs.uni-bonn.de/backend/v1/files/attachments/stylefurniture_supp.pdf)) consists of ~90K furniture items, each containing:
- An **image**
- **Metadata** (Brand, Dimensions, Weight, Color, Material, Category, Style)

### Approaches
Two embedding approaches were evaluated:

**1. Image + Metadata (Approach 1)**
- **Image Embedding**: CLIP (`openai/clip-vit-base-patch16`)
- **Metadata Embedding**: MiniLM (`all-MiniLM-L6-v2`)

**2. Image + Metadata + Description (Approach 2)**
- Same as **Approach 1**, but with an additional **LLM-generated description**.
- **Description Generation**: `ollama` (`llama3.2-vision:11b-instruct-q8_0`)
- **Description Embedding**: MiniLM (`all-MiniLM-L6-v2`)

### Metric and Graph Selection

The metrics were chosen to evaluate embedding quality across complementary dimensions:  

1. **Category Consistency**  
   - Measures how well the recommendation system keeps similar items together. If it works well, when you search for a chair, you mostly get other chairs — not sofas or tables. If the consistency is low, the system is confused and mixes up different categories. Higher is better.
 
2. **KL Divergence**  
   - Quantifies distributional differences between embedding approaches.  

3. **Similarity Statistics** (Mean/Variance)  
   - Shows the mean and variance of scores for queries. 
      - Higher mean =  Model has higher confidence in its findings.
      - Lower Variance = Model is more consistent.

4. **High-Quality Matches** (>0.8 similarity)  
   - How many matches have score > Threshold [1 being perfect match].  

5. **Nearest Neighbor Consistency** (Jaccard)  
   - Jaccard Similarity based on indices i.e. How many neighbors are same. [1 - Perfect, 0 - Random]  

6. **Intraclass Distance**  
   - This tells us how close together similar items are in their group. If chairs are very close to other chairs (small intraclass distance), it means the system understands that they are alike. If they are far apart (large intraclass distance), then the system struggles to group similar things. Lower is better. Here we have averaged it over the classes.

7. **t-SNE Plot Analysis**  
   - Shows how the system organizes furniture in 2D space.  
     - Clear clusters = The system groups similar items well (e.g., chairs with chairs, tables with tables).  
     - Mixed points = The system struggles to distinguish categories.  

8. **Similarity Histogram**  
   - Tells us how confident the system is in its recommendations:  
     - Tall bars on the right (near 1.0) = Many high-confidence matches.  
     - Even spread = Recommendations vary in quality.  
   
9. **Similarity Score Distribution by Category [Box Plot]**  
   - Compares recommendation strength category-by-category:  
     - Higher median scores = More reliable matches for that category.  
   
10. **Category-wise Similarity Comparison [Bar Graph]**  
   - Measures how well items *within the same category* match each other:  
     - Higher average similarity = Better category understanding.  
   


### Results
<table>
  <tr>
    <th>Metric</th>
    <th>Approach 1</th>
    <th>Approach 2</th>
  </tr>
  <tr>
    <td><strong>Category Consistency</strong></td>
    <td>0.9602</td>
    <td>0.9649</td>
  </tr>
  <tr>
    <td ><strong>KL Divergence</strong></td>
    <td colspan="2" align="center"> 0.0031</td>
  </tr>
  <tr>
    <td><strong>Mean Similarity</strong></td>
    <td>0.9559</td>
    <td>0.9548</td>
  </tr>
  <tr>
    <td><strong>Similarity Variance</strong></td>
    <td>0.0009</td>
    <td>0.0009</td>
  </tr>
  <tr>
    <td><strong>High-Quality Matches (>0.8)</strong></td>
    <td>99.78%</td>
    <td>99.78%</td>
  </tr>
  <tr>
    <td ><strong>Avg. NN Consistency (Jaccard)</strong></td>
    <td colspan="2" align="center"> 0.9102</td>
  </tr>
  <tr>
    <td><strong>Intraclass Distance</strong></td>
    <td>0.8454</td>
    <td>0.8452</td>
  </tr>
</table>


- **t-SNE** - Both approaches show similar clustering patterns.  
- **Similarity Histogram** - Both approaches produce equally confident matches (>99% high-quality).  
- **Similarity Score Distribution by Category** - Approach 1 (without descriptions) scores slightly higher in all categories.  
- **Category-wise Similarity Comparison**  - Approach 1 consistently outperforms Approach 2.  


**Graphs** are available in `graphs/evaluation/`.  
**Full results** are logged in `evaluation.txt`.

### Files Used
1. **Dataset Preparation**  
   - `make_full_dataset.py`: Extracts image paths and metadata.  
   - Generates dataset with Columns: `Image Path, Brand, Dimensions [W X D X H], Weight, Color, Material, Category, Style`.

2. **Description Generation (Approach 2 only)**  
   - `ollama_generate_description.py`: Uses Ollama (llama3.2-vision) to generate image descriptions.

3. **Embedding Generation**  
   - `make_embeddings.py`: Embeds images (CLIP), metadata (MiniLM), and descriptions (MiniLM).

4. **Final Dataset Creation**  
   - `create_dataset.py`: Compiles embeddings into two datasets (`clip_metadata_embed.csv`, `clip_metadata_desc_embed.csv`).

5. **Evaluation**  
   - `evaluation.py`: Computes metrics and generates graphs.



### Ablation Study: Evaluating Component Influence

#### Objective
Determine the individual contribution of each component (image embeddings, metadata, descriptions) to recommendation quality, while investigating potential CLIP overfitting.


#### Models Used
| Embedding Type | Model                   | Rationale                                                                 |
|----------------|-------------------------|---------------------------------------------------------------------------|
| Image (CLIP)   | `openai/clip-vit-base-patch16` | Standard baseline, but potential overfitting risk (trained post-dataset)  |
| Image (DINOv2) | `dinov2_vits14`         | Vision transformer trained without text supervision (mitigates overfitting)|
| Text           | `all-MiniLM-L6-v2`      | Lightweight but effective metadata/description encoder                    |

#### Tested Cases
```python
[
    "Baseline_clip (Image_clip + Metadata)",
    "Baseline_dino (Image_dino + Metadata)", 
    "CLIP Image (Image-only)",
    "DINO Image (Image-only)",
    "No Metadata (Image_clip + Text)",
    "No Metadata (Image_dino + Text)",
    "Approach 2 (Image_clip + Metadata + Text)",
    "Approach 2 (Image_dino + Metadata + Text)"
]
```

#### Implementation
1. **Embedding Generation**  
   - DINOv2 embeddings:  
     - Batch processing: `make_abalation_image_embeddings_DINOv2.py`  
     - Aggregation: `combine_dino_embeddings.py`  
   - CLIP embeddings: Reused from main pipeline  

2. **Evaluation**  
   - Script: `ablation_study.py`  
   - Metrics:  
     - Mean Similarity  
     - Variance  
     - Median Similarity  
   - Visualizations:  
     - 3 Bar plots (mean/variance/median)  
     - 1 Box plot (mean similarity distribution)  
     - *Location*: `graphs/ablation/`  

#### Key Findings
1. **Image Embeddings Comparison**  
   - CLIP vs. DINOv2 performance delta indicates overfitting likelihood  
   - CLIP significantly outperforms DINOv2  
   - CLIP when combined with other embeddings also performs worse.

2. **Component Contribution**  
   - Metadata impact: `Image + Metadata` performs better than `Image + Description` in the case of CLIP and there is no significant difference in case of DINOv2 embedding.
   - Description impact: `Image + Description` performs better than `Image + Metadata + Description` in the case of CLIP and there is no significant difference in case of DINOv2 embedding.

3. **Stability Analysis**  
   - Variance metrics reveal robustness of each configuration 
   - Low Variance for all cases indicates highly consistent recommendations. Regardless of image encoder selection or text inclusion.
   - The 3-order-of-magnitude lower variance in CLIP embeddings (10⁻⁶ vs DINOv2's 10⁻³) suggests potential overfitting to this dataset.


### Overfitting Check: CLIP vs. DINOv2 Embeddings

#### Objective
Determine whether CLIP embeddings exhibit overfitting compared to DINOv2 by analyzing:  
1. Similarity score distributions  
2. Neighbor diversity in k-NN searches  

#### Methodology 
(`clip_overfitting.py` implements the following pipeline)  

1. **Data Preparation**  
   - Load normalized embeddings:  
     ```python
     CLIP: clip_embeddings.npy  
     DINOv2: image_embeddings_DINOv2_combined.npy
     ```
   - L2-normalize all embeddings for cosine similarity comparison  

2. **k-NN Analysis** (k=5)  
   - Build FAISS indices for efficient similarity search  
   - Compute pairwise similarity scores between items  

3. **Metrics**  
   - **Similarity Distribution**: Density plot of scores (0=dissimilar, 1=identical)  
   - **Neighbor Diversity**: Ratio of unique neighbors per item (1.0 = all neighbors are distinct)  

#### Key Results
| Metric                  | CLIP       | DINOv2     | Interpretation                     |
|-------------------------|------------|------------|------------------------------------|
| **Neighbor Diversity**  | 1.00       | 1.00       | Both models avoid collapsing to identical neighbors |
| **Similarity Scores**   | [See Plot] | [See Plot] | CLIP shows tighter score distribution |


The KDE (Kernel Density Estimation) plot compares the **distribution of similarity scores** between image-only embeddings generated using **CLIP** and **DINOv2** models.

#### Interpretation:

- **CLIP Embeddings (Blue):**
  - Display a **sharply peaked distribution** near the upper bound (~1.0), indicating a **large concentration of very high similarity scores**.
  - The lower subplot (rug plot) shows fewer low or mid-range similarity scores, suggesting a **lack of score diversity**.
  - This sharp skew towards high similarity implies that CLIP might be **overfitting**—i.e., it assigns overly confident similarity scores even for dissimilar or only loosely related items.
  
- **DINOv2 Embeddings (Red):**
  - Show a **smoother, broader distribution** of similarity scores, including mid-range values.
  - This suggests DINOv2 provides more **nuanced and discriminative similarity scores**, better capturing subtle differences between items.




#### Understanding KDE (Kernel Density Estimation) Plots

A **KDE plot** is a smoothed version of a histogram used to estimate the **probability density function** of a continuous variable. It helps visualize how values are distributed across a dataset.

##### Key Elements of a KDE Plot:

- **X-axis:** Represents the range of values for the variable (e.g., similarity scores).
- **Y-axis:** Represents the **estimated density**—not raw counts, but how frequently values appear, normalized to sum to 1.
- **Curve:** The height of the curve at any point indicates the **relative likelihood** of seeing that value.

##### How to Read a KDE Plot:

1. **Peak Height:**
   - A **taller peak** means that values in that range appear **more frequently**.
   - A **flatter or wider** curve suggests values are more **spread out** (i.e., more diverse).

2. **Width and Shape:**
   - **Narrow, high peaks** → Concentrated values, low diversity, potential overconfidence or overfitting.
   - **Broader peaks** → More varied distribution, indicating more nuanced behavior.

3. **Comparison Between Lines:**
   - When comparing two KDE curves (e.g., CLIP vs. DINOv2), look at the **differences in location, height, and spread**.
   - If one of the model’s curve is too sharply peaked near 1.0, it might be **overestimating similarity**, losing discriminatory power.

4. **Rug Plot:**
   - The small vertical ticks below the KDE curve show the **actual data points**, helping the reader connect the density curve to the underlying samples.



##### Why It Matters:
In tasks like similarity-based recommendation, KDE plots reveal whether the used model is assigning scores in a balanced way or leaning too heavily toward extreme values. Overconfident similarity estimates can lead to poor generalization and degraded performance on unseen or borderline cases.



#### Conclusion:

This comparison indicates that while CLIP may yield high performance on surface metrics, it risks **overfitting to the dataset**, reducing generalization. In contrast, DINOv2 produces a more balanced score distribution, making it a better candidate for capturing **fine-grained similarity** in visual recommendation tasks.

## Utility Scripts

### `remove_duplicates.py`  
**Purpose**: Removes near-duplicate images (CLIP similarity > 0.99) to prevent dataset bias.  

**Key Steps**:  
1. Loads embeddings (CLIP/DINOv2/metadata/text) and CSV  
2. Uses FAISS to flag duplicates via cosine similarity  
3. Deletes duplicate images and saves cleaned data  

**Usage**:  

The threshold parameter can be changed as per requirement.
```bash
python remove_duplicates.py
```
**Outputs**:  
- `*_dedup.npy` (cleaned embeddings)  
- `*_dedup.csv` (filtered metadata)  
- Logs removed duplicates (e.g., `5,500/90,000` items)  


### `make_demo_dataset.py`

**Purpose**: Creates a smaller demo dataset from sampled images, generating:  
- CLIP image embeddings  
- LLM-generated descriptions  
- Text embeddings of descriptions  

**Workflow**:  
1. **Input**: PNG images from `dataset/dataset6/data/`  
2. **Processing**:  
   ```python
   for each image:
       image_embedding = CLIP(image)  # Uses `generate_image_embedding()`
       description = llama(image)     # Uses `generate_description()`
       text_embedding = MiniLM(description)  # Uses `generate_text_embedding()`
   ```  
3. **Outputs**:  
   - `demo_dataset.csv`: Image paths + descriptions  
   - `demo_embeddings.npy`: Combined image+text embeddings  

**Usage**:  
```bash
python make_demo_dataset.py
```  

**Key Features**:  
- Processes every image 
- Integrates with existing embedding pipelines  
- Saves embeddings in same format as main dataset  


### `make_dino_demo_dataset.py`

**Purpose**:  
Essentially same as `make_demo_dataset.py` but uses DINOv2 embeddings instead of CLIP, enabling direct comparison of vision models in the recommendation system. Important as CLIP might be overfitting on the given dataset.



### Demo Application (`demo_rec_app.py`) - Lightweight Streamlit Demo

**Key Differences from Full App**:
- Uses the smaller `demo_embeddings.npy` dataset  
- Shows generated descriptions   
- Simplified interface for quick testing  

**Features**:
```python
1. Same 4 distance metrics (L2/Cosine/Manhattan/Chebyshev)
2. FAISS-powered similarity search
3. Displays:
   - Query image + generated description
   - Top-k matches from main dataset
   - Visual similarity scores
```

**Usage**:
```bash
streamlit run demo_rec_app.py
```

**Implementation Notes**:
- Queries against full dataset but uses demo set as input  
- Maintains identical distance metric implementations  

**Why This Exists**:
- Enables testing without loading full 90K embeddings  
- Showcases description generation quality  
--- 
