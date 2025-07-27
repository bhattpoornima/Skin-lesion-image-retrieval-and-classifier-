import streamlit as st
from PIL import Image
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, Model
import matplotlib.pyplot as plt


st.title("🩺 SkinAI: Disease Detector & Similarity Finder")
uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

# import gdown
# import zipfile

# Only download if not already present
if not os.path.exists("HAM10000_metadata.csv"):
    url = "https://drive.google.com/uc?id=19Cbv_ZDMr1rJIXaWobCe_7_5bfkf4ImY"  # Replace with actual ID
    output = "data.zip"
    gdown.download(url, output, quiet=False)

    # Unzip it
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall()

# Load all necessary files (these must exist in your project folder)
db_hashes = np.load("hash_codes_binary.npy")
# Fix the paths from Colab to local
from pathlib import Path

db_filenames = np.load("filenames.npy", allow_pickle=True)
db_filenames = [str(Path("merged_images") / "HAM10000_merged" / Path(f).name) for f in db_filenames]


metadata_df = pd.read_csv("HAM10000_metadata.csv")

# Prepare metadata dictionary
metadata_df['image_id'] = metadata_df['image_id'].astype(str)
metadata_dict = metadata_df.set_index('image_id').to_dict(orient='index')



if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    resized_image = uploaded_image.resize((300, 300))
    st.image(resized_image, caption='Uploaded Image', use_container_width=False)

    with st.spinner("Analyzing image and retrieving similar cases..."):
        # Load the Sequential model
        model = load_model("ham10000_model_200epochs.keras")

        # Extract output from the second dense layer (change index as needed)
        target_layer_name = "dense_2"  # Update this if needed

        # Now define the feature extractor using layer index (safer than names)
        feature_extractor = Model(inputs=model.layers[0].input,
                                outputs=model.get_layer(target_layer_name).output)

        def binarize_features(features):
                return np.where(np.tanh(features) > 0, 1, 0)


        # Load query image
        query_img = uploaded_image.resize((64, 64))  # Resize to match training input size
        query_arr = image.img_to_array(query_img) / 255.0
        query_arr = np.expand_dims(query_arr, axis=0)
        # hashing the query image
        query_feat = feature_extractor.predict(query_arr)
        query_hash = binarize_features(query_feat)

        # reversing the predicted idx to the appropriate class
        import json

        # Load the saved class_indices
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)

        # Reverse it for index to class mapping
        index_to_class = {v: k for k, v in class_indices.items()}


        # 🧠 Step 1: Predict class
        pred = model.predict(query_arr)
        predicted_class_idx = np.argmax(pred)
        predicted_class_name = index_to_class[predicted_class_idx]
        st.success(f"✅ Predicted Disease Class: {predicted_class_name}")



        def hamming_distance(a, b):
            return np.sum(a != b)


        # Compute distances
        distances = [hamming_distance(query_hash[0], db_hash) for db_hash in db_hashes]
        top_k_indices = np.argsort(distances)[:5]

    # Display query image again with prediction
    st.image(resized_image, caption=f"Query Image - Predicted: {predicted_class_name}", width=200)

    # Show top-5 similar images
    st.subheader("🔍 Top 5 Similar Images (Based on Hamming Distance)")

    cols = st.columns(5)
    for i, idx in enumerate(top_k_indices):
        filepath = db_filenames[idx]
        img_id = os.path.splitext(os.path.basename(filepath))[0]  # ID without extension
        sim_img = image.load_img(filepath, target_size=(64, 64))

        meta = metadata_dict.get(img_id, {})
        dx = meta.get('dx', 'N/A')
        age = meta.get('age', 'N/A')
        sex = meta.get('sex', 'N/A')
        loc = meta.get('localization', 'N/A')

        with cols[i]:
            st.image(
                sim_img,
                caption=(
                    f"Diagnosis: {dx}\n"
                    f"Age: {age}\n"
                    f"Sex: {sex}\n"
                    f"Location: {loc}\n"
                    f"Distance: {distances[idx]}"
                ),
                width=150
            )
