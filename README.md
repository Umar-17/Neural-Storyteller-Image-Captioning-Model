# Neural Storyteller - Image Captioning with Seq2Seq

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)


A Generative AI project that generates descriptive captions for images using a Sequence-to-Sequence (Seq2Seq) encoder-decoder architecture. The model leverages a pre-trained ResNet50 for image feature extraction and an LSTM-based decoder for text generation.

## 📌 Project Overview

This project implements an image captioning system capable of understanding visual content and translating it into natural language descriptions. It features a complete pipeline from data preprocessing to model training and deployment via a Streamlit web application.

### Key Features
*   **Encoder**: Pre-trained **ResNet50** (with the classification head removed) to extract high-level image features.
*   **Decoder**: **LSTM** (Long Short-Term Memory) network with embedding layers to generate captions word-by-word.
*   **Search Algorithms**: Implements both **Greedy Search** and **Beam Search** for caption generation during inference.
*   **Interactive App**: A user-friendly **Streamlit** interface to upload images and generate captions in real-time.

## � Languages & Tools

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

## �📂 Project Structure

```
├── Model/                  # Directory for saving model checkpoints
├── Notebook/
│   └── genai-assignment-1.ipynb  # Main Jupyter notebook for training and evaluation
├── app.py                  # Streamlit web application for inference
├── README.md               # Project documentation
└── requirements.txt        # (Optional) List of dependencies
```

## 📊 Dataset

The model is trained on the **Flickr30k** dataset, which consists of 31,000+ images, each annotated with 5 descriptive captions.

> **Note**: The dataset path in the notebook corresponds to a Kaggle environment (`/kaggle/input/...`). You may need to adjust paths if running locally.

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   PyTorch
*   Torchvision
*   Streamlit
*   Pillow
*   NLTK
*   Pandas
*   Scikit-learn
*   Tqdm

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd "Neural Storyteller - Image captioning with seqtoseq"
    ```

2.  Install dependencies:
    ```bash
    pip install torch torchvision streamlit pillow nltk pandas scikit-learn tqdm
    ```

## 🛠️ Usage

### 1. Training the Model
To train the model from scratch, open and run the Jupyter Notebook:
`Notebook/genai-assignment-1.ipynb`

The notebook covers:
1.  **Feature Extraction**: Extracting and saving ResNet50 features for all images.
2.  **Vocabulary Building**: Tokenizing captions and building a vocabulary.
3.  **Model Definition**: Defining the Encoder (Linear) and Decoder (LSTM).
4.  **Training Loop**: Training the model with CrossEntropyLoss.
5.  **Evaluation**: Calculating BLEU-4, Precision, Recall, and F1 scores.
6.  **Saving Artifacts**: Saves `model_weights.pth` and `vocab.pkl` for the app.

### 2. Running the Web App
Ensure the trained model weights (`model_weights.pth`) and vocabulary (`vocab.pkl`) are placed in the `Model/` directory. Then, launch the interactive app:

```bash
streamlit run app.py
```

*   **Upload an Image**: Supports JPG and PNG formats.
*   **Select Method**: Choose between *Greedy Search* (faster) or *Beam Search* (typically higher quality).
*   **View Result**: The model will generate and display a caption for your image.

## 🧠 Model Architecture

### Encoder
*   **Base**: ResNet50 (pre-trained on ImageNet).
*   **Transformation**: The last fully connected layer is removed.
*   **Adaptation**: A linear layer transforms the 2048-dimensional ResNet output to the embedding size (e.g., 512).

### Decoder
*   **Embedding Layer**: Converts word indices to dense vectors.
*   **LSTM Layer**: Takes the concatenated image features (at step 0) and word embeddings to predict the next word in the sequence.
*   **Linear Output**: Maps LSTM hidden states to vocabulary size scores.

## 📈 Performance Metrics

The project evaluates generated captions using standard NLP metrics:
*   **BLEU-4**: Measures n-gram overlap between generated and reference captions.
*   **Precision & Recall**: Evaluates the relevance of generated words.
*   **F1-Score**: Harmonic mean of precision and recall.

## 🤝 Acknowledgements

*   Dataset provided by **Flickr30k**.
*   Pre-trained models from **Torchvision**.

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

