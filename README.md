# Toxicity Prediction using Deep Learning

This project was developed as part of the *Computer Aided Drug Design (CADD)* course. The goal was to predict molecular toxicity using SMILES representations and deep learning models.

## 📌 Project Overview
Toxicity prediction plays a critical role in drug discovery. Our project uses the Tox21 dataset to train and evaluate deep learning models that classify whether molecules are toxic or non-toxic based on their SMILES representation.

We explore multiple deep learning architectures, including:
- **RNN (Recurrent Neural Network)**
- **CNN (Convolutional Neural Network)**
- **MCA (Multiscale Convolutional Attention)**
- **Dense Feedforward Networks**

All models are trained using tokenized SMILES strings and evaluated using classification metrics such as F1 Score, ROC-AUC, and Precision-Recall.

## 📁 Repository Structure
```
CADD_PROJECT_CODE/
├── data/                     # Raw dataset and preprocessed files
├── output/                   # Saved results, model weights, tokenizer files
├── params/                   # Model hyperparameters
├── public/trained_models/   # Public trained models (v0)
├── scripts/                 # Scripts to train and evaluate models
├── toxsmi/                  # Main Python package with models and utils
├── requirements.txt         # Required Python packages
├── setup.py                 # Setup script
└── README.md                # Project documentation
```

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/CADD_PROJECT_CODE.git
cd CADD_PROJECT_CODE
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train a model
```bash
python scripts/model_train.py --params params/mca.json
```

### 4. Evaluate a model
```bash
python scripts/model_evaluation.py --params params/mca.json
```

## 📊 Results

Evaluation metrics are saved under `output/debug/results/`. Models are saved in `output/debug/weights/` and can be loaded for reuse.

### Sample Metrics
- **F1 Score**: 0.81
- **ROC-AUC**: 0.89
- **Precision-Recall AUC**: 0.86

## 🧠 Key Concepts

- **SMILES Tokenization**: Custom tokenizer for molecular strings.
- **Chemical Representation Learning**: Learned embeddings via VAE and deep models.
- **Model Comparison**: Performance assessed across MCA, RNN, CNN, and Dense architectures.

## 📂 Dataset

- **Tox21 Dataset**  
  Contains molecular SMILES and toxicity labels for multi-label classification tasks.  
  Source: [https://tripod.nih.gov/tox21/challenge/](https://tripod.nih.gov/tox21/challenge/)

## 🔧 Parameters and Models

Hyperparameters for models are defined in JSON under the `params/` directory. Trained model weights are available under `output/debug/weights/` and `public/trained_models/`.

## 🤝 Contributors

- **Nikhil Kumar** (2022322)  
- **Aditya Kumar Sinha** (2022034)  
- **Nutan Kumari** (2022341)  
- **Harsh Vishwakarma** (2022205)


> Developed as part of the *Computer Aided Drug Design* course project.
