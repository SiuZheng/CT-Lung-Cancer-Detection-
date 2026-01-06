# CT Lung Cancer Detection

## 📌 Project Overview
This project implements a Deep Learning approach for the detection of lung cancer from CT scan slices. It explores advanced feature engineering techniques and hybrid model architectures to classify CT scan images into three categories: **Benign**, **Malignant**, and **Normal**.

The core of the project involves a clear comparison between a standard **EfficientNetV2** baseline and a custom hybrid model termed **HCTLFN** (Hybrid Convolutional-Transformer-Like Feature Network), which utilizes a specialized preprocessing module called **AGOC** (Adaptive Gamma Optimization Correction).

## 📊 Dataset
The project utilizes the **The IQ-OTHNCCD lung cancer dataset** available on Kaggle.
- **Source**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset)
- **Classes**:
  - `Malignant`: Cancerous cases
  - `Benign`: Non-cancerous tumor cases
  - `Normal`: Healthy cases
- **Preprocessing**: 
  - The dataset is automatically downloaded and reorganized into `train` and `test` directories.
  - Class balancing is handled using computed class weights during training.

## 🛠 Methodology

### 1. Feature Engineering: AGOC (Adaptive Gamma Optimization Correction)
A custom preprocessing layer designed to enrich the input features.
- **Input**: Standard 3-channel RGB images.
- **Output**: 5-channel tensors.
- **Components**:
  - Original 3 RGB channels.
  - 1 Grayscale channel (weighted sum).
  - 1 Edge channel (Sobel filter magnitude).
- **Normalization**: Applies standard normalization to the 5-channel stack.

### 2. Model Architectures

#### **HCTLFN (Hybrid Model)**
A sophisticated architecture designed to capture both spatial and sequential features:
1.  **Input Layer**: Accepts 5-channel AGOC inputs.
2.  **Backbone**: **EfficientNetV2-S** (pre-trained) used as a feature extractor.
3.  **Tokenization**: Convolutional reduction to create a sequence of feature "tokens".
4.  **Sequential Processing**: **Bi-directional GRU** (BiGRU) to capture global context across the feature map.
5.  **Classification Head**: Dropout + Linear layer.

#### **EfficientNetClassifier (Baseline)**
A standard implementation for benchmarking:
- **Backbone**: **EfficientNetV2-S** (pre-trained).
- **Pooling**: Global Average Pooling.
- **Classifier**: Standard linear head.

## ⚙️ Usage

### Prerequisites
The project requires Python and the following libraries:
- `torch`, `torchvision` (PyTorch ecosystem)
- `timm` (PyTorch Image Models)
- `scikit-learn` (Metrics & Utils)
- `matplotlib` (Visualization)
- `tqdm` (Progress bars)
- `kagglehub` (Dataset downloading)

### Running the Project
1.  **Open the Notebook**: Launch `lung_cancer.ipynb` in a Jupyter environment or Google Colab.
2.  **Download Data**: The first cell automatically downloads the dataset from Kaggle using `kagglehub`.
3.  **Training**: Run the training cells. The notebook trains both the `HCTLFN` and `EfficientNet` models using the defined training loop with Early Stopping.
4.  **Evaluation**: The training history (Accuracy/Loss) is plotted at the end for comparison.

## 📈 Results
The models are evaluated based on Validation Accuracy and Loss.
- **EfficientNetV2 Baseline**: showed strong convergence, achieving high validation accuracy (reaching ~97%+ in experiments).
- **HCTLFN**: demonstrates the capability of integrating edge and grayscale features, achieving ~91%+ accuracy with a more complex architecture and generalize more well compared to the original basemodel.

*(Note: Exact results may vary based on random seed and split)*

## 📂 File Structure
- `lung_cancer.ipynb`: Main project notebook containing all code for data loading, model definition, training, and evaluation.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hYBJzQ4_ye_YOasWDA6nSLkFmPH_VDiR?usp=sharing)
- `README.md`: Project documentation.
