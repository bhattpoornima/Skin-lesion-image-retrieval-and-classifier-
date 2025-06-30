# Skin Lesion Image Retrieval System using Deep Learning

A deep learning–based image retrieval and classification system built on the HAM10000 dataset to assist in the automated analysis of pigmented skin lesions. The project combines CNN-based feature extraction with binary hashing and Hamming similarity to retrieve visually and diagnostically similar cases.

---

## 📌 Motivation

Training of neural networks for automated diagnosis of pigmented skin lesions is hampered by the small size and lack of diversity of available dermatoscopic image datasets. To address this, our system not only classifies the lesion image into one of seven diagnostic categories but also retrieves the top visually similar lesion images from a reference set, providing interpretable decision support for clinical use.

---

## 🧠 Project Highlights

* Trained a custom Convolutional Neural Network (CNN) to classify dermatoscopic images into **7 diagnostic categories**:

  * `akiec` – Actinic keratoses and intraepithelial carcinoma
  * `bcc` – Basal cell carcinoma
  * `bkl` – Benign keratosis-like lesions
  * `df` – Dermatofibroma
  * `mel` – Melanoma
  * `nv` – Melanocytic nevi
  * `vasc` – Vascular lesions
* Extracted deep features using intermediate dense layers from the trained model.
* Converted these feature vectors to binary hash codes via `tanh` activation and thresholding.
* Computed **Hamming distance** to find the top-k visually similar lesions from the dataset.
* Retrieved images are visualized along with metadata (diagnosis, age, sex, location).

---

## Results

(k=5)

Precision@k: 0.8000
Recall@k: 1.0000
F1@k: 0.8889
Accuracy@k: 0.8000
Hit@k: 1.0000
AP: 0.8042

![image](https://github.com/user-attachments/assets/1a6bf24c-5c1b-4fb7-9fe5-69deeaa1a37b)

![image](https://github.com/user-attachments/assets/e6e5c62e-7bef-4d00-9468-246eabaafd4f)


## Code

Recruiters and faculty may request access via email.
