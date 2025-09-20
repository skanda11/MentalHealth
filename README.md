# AI-Powered Mental Distress Trend Analyzer

## Project Title: An AI-Powered System for Temporal Trend Analysis of Mental Distress Indicators in Online Text



---

## 💡 Overview

This project develops an AI-driven system to identify early indicators of mental distress by analyzing patterns and trends in anonymized online text. Unlike traditional methods that classify individual posts, this system focuses on the *temporal evolution* of sentiment, providing a more nuanced and potentially earlier warning of escalating distress. The pipeline is designed to be modular and configurable, allowing for easy modification and experimentation.

**Key Features:**
* **Configurable Pipeline:** All parameters managed via a central `config.yaml` file.
* **Modular Design:** Code separated into distinct, manageable directories (data ingestion, preprocessing, classification, temporal analysis, dashboard).
* **State-of-the-Art NLP:** Utilizes a fine-tuned `DistilBERT` model for accurate text classification.
* **Temporal Trend Analysis:** Implements a sliding window and moving average approach to detect sustained patterns of distress.
* **Interactive Dashboard:** A `Streamlit` application for visualizing trends and identifying periods of high concern.

## 🎯 Problem Statement

Mental health conditions often go undiagnosed due to stigma and lack of awareness. While individuals frequently express their feelings online, manually sifting through vast amounts of text for early warning signs is impractical. Standard AI models typically classify isolated text snippets, missing the crucial context of how an individual's sentiment evolves over time. This project addresses this gap by providing a system that tracks and highlights persistent negative shifts in textual expression.

## 🚀 Project Architecture

The system follows a sequential data pipeline:

```mermaid
graph LR
    A[Anonymized Text Posts (Input)] --> B{Preprocessing: Cleaning & Tokenization};
    B --> C[Distress Classifier: Fine-tuned DistilBERT];
    C -- Distress Probability Score (0.0 - 1.0) per Post --> D[Temporal Analyzer: Custom Python Module];
    D -- Smoothed Trend Score (e.g., Moving Average) --> E[Interactive Dashboard: Streamlit];
    E -- Visualizes Trend & Alerts --> F[User/Platform (Output Insights)];
    style A fill:#ADD8E6,stroke:#333,stroke-width:2px,color:#000
    style B fill:#F9D05A,stroke:#333,stroke-width:2px,color:#000
    style C fill:#90EE90,stroke:#333,stroke-width:2px,color:#000
    style D fill:#FFB6C1,stroke:#333,stroke-width:2px,color:#000
    style E fill:#ADD8E6,stroke:#333,stroke-width:2px,color:#000
    style F fill:#D3D3D3,stroke:#333,stroke-width:2px,color:#000
```

## 🛠️ Technologies Used

* **Programming Language:** Python 3.9+
* **Machine Learning/NLP:** PyTorch, Hugging Face Transformers (`DistilBERT`), Hugging Face `datasets`, Scikit-learn, NLTK
* **Data Handling:** Pandas, NumPy
* **Configuration:** YAML (`PyYAML`)
* **Visualization/Frontend:** Streamlit, Plotly
* **Kaggle API:** For dataset ingestion
* **Orchestration:** Custom Python scripts

## ⚙️ Setup and Installation

### 1\. Clone the Repository

```bash
git clone <your-repository-url>
cd mental_distress_pipeline
```

### 2\. Create Virtual Environment (Recommended)

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt` content:**
```
pandas
pyyaml
numpy
scikit-learn
nltk
transformers
datasets
torch
streamlit
plotly
kaggle
```

*If you have a CUDA-enabled GPU, you might need a specific PyTorch installation. Refer to [PyTorch's official website](https://pytorch.org/get-started/locally/) for instructions matching your system (e.g., for CUDA 11.8: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`).*

### 4\. Kaggle API Configuration

To download the dataset, you need to set up your Kaggle API credentials:

* Go to Kaggle, log in, and navigate to your account settings.
* Click "Create New API Token" to download `kaggle.json`.
* Place this `kaggle.json` file in the appropriate directory:
    * **Linux/macOS:** `~/.kaggle/`
    * **Windows:** `C:\Users\<Your-Username>\.kaggle\`
* Ensure the `.kaggle` directory and `kaggle.json` file have appropriate permissions (e.g., `chmod 600 ~/.kaggle/kaggle.json` on Linux/macOS).

### 5\. NLTK Data Download

Your preprocessing script requires specific NLTK data. Run the following once:

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"
```

## 🚀 Usage

### 1\. Configure Parameters

All pipeline parameters are defined in `config/config.yaml`. Before running, review this file and adjust any paths or hyperparameters as needed.

* `data_ingestion`: Kaggle dataset name, download path.
* `preprocessing`: Raw/processed data paths, text/target columns, train/test split ratio.
* `classification`: Model name, max sequence length, batch size, epochs, learning rate, save paths.
* `temporal_analysis`: Input/output paths, moving average period, alert threshold, consecutive alerts required.
* `dashboard`: Input data for visualization.

### 2\. Run the Entire Pipeline

From the root directory of the project (`mental_distress_pipeline/`), execute the `main.py` script:

```bash
python main.py
```

This script will sequentially:

1.  Download the Kaggle dataset.
2.  Preprocess the raw text data.
3.  Train a `DistilBERT` classification model.
4.  Make distress probability predictions on the entire dataset.
5.  Perform temporal trend analysis, calculating smoothed trends and identifying alerts.

*Note: The model training step (Step 3) can be time-consuming, especially without a GPU.*

### 3\. View the Interactive Dashboard

Once the `main.py` script completes, you can launch the Streamlit dashboard to visualize the results:

```bash
streamlit run dashboard/app.py
```

This will open the dashboard in your default web browser, displaying the distress probability trends and any detected periods of high concern.

## 📊 Results

* **Distress Classification Model (Fine-tuned DistilBERT):**
    * **Accuracy:** 
    * **Precision (Distress Class):** 
    * **Recall (Distress Class):** ** **
    * **F1-Score (Distress Class):** ** **
* **Temporal Analysis:** The system successfully demonstrated its ability to smooth out individual post-level fluctuations and highlight sustained negative trends over time, providing a more robust indicator than single-post classification.

## 🤝 Contributing

This project is part of an M.Tech curriculum. While direct contributions are not solicited, suggestions and feedback are welcome!

