# 🧪 Toxicity Prediction using Deep Learning

This project was developed as part of the **Computer Aided Drug Design (CADD)** course. It focuses on predicting chemical compound toxicity using modern deep learning techniques applied to SMILES representations. The project builds on recent advancements in **chemical language modeling** and **attention-based neural networks** to help identify toxic molecules early in the drug development process.

---

## 📌 Project Overview

Toxicity is one of the leading causes of failure in clinical drug development. Traditional **in vitro** and **in vivo** assays are time-consuming and expensive. This project applies **data-driven, AI-based approaches** to predict molecular toxicity more efficiently and scalably.

Key objectives:
- Explore deep learning for toxicity prediction using SMILES strings.
- Compare multiple model architectures (RNN, CNN, Dense, MCA).
- Use data augmentation and uncertainty estimation for robust learning.
- Train and evaluate models on the Tox21 dataset.

---

## 🧬 Dataset

**📁 Source**: [Tox21 Challenge Dataset](https://tripod.nih.gov/tox21/challenge/)  
**🧪 Description**: Multi-label dataset with ~12,000 molecules represented as SMILES strings. Labels include toxic effects such as:
- Nuclear receptor signaling disruption
- Stress response pathways

### ✅ Preprocessing:
- SMILES Standardization
- Invalid entries removal
- 80/10/10 train/validation/test split
- Atom-order randomization for SMILES augmentation

---

## ⚙️ Model Architectures

### 🔸 MCA (Multiscale Convolutional Attention)
- Input: Randomized SMILES → tokenized
- Multi-kernel 1D CNNs (kernel sizes: 3, 5, 11)
- Self-attention layers for substructure focus
- Dense layers for final prediction

### 🔹 RNN (Bi-LSTM)
- Bidirectional LSTM layers
- Captures sequential dependencies in SMILES
- Tuned with 10 layers, dropout=0.5

### 🔹 CNN
- Stacked 1D convolutions
- Simple yet effective for capturing local patterns

### 🔹 Dense
- Fully connected layers
- Used as a baseline for performance comparison

---

## 📊 Results

| Model     | ROC-AUC | F1 Score | PR-AUC | Accuracy | Notes                             |
|-----------|---------|----------|--------|----------|-----------------------------------|
| MCA       | 0.840   | 0.39     | 0.71   | 0.73     | Strong generalization             |
| Fine-tuned RNN | 0.8677  | 0.10     | 0.76   | 0.63     | Improved with tuning + dropout    |
| CNN       | 0.801   | 0.28     | 0.65   | 0.69     | Fast and lightweight              |
| Dense     | 0.764   | 0.22     | 0.61   | 0.66     | Baseline                          |

> 📂 All evaluation metrics and plots are saved in `output/debug/results/`.

---

## 🔄 Training & Evaluation

### Training
```bash
python scripts/model_train.py --params params/mca.json
```

### Evaluation
```bash
python scripts/model_evaluation.py --params params/mca.json
```

### Hardware
- GPU: NVIDIA A400
- Batch Size: 64
- Epochs: 200
- Learning Rate: 1e-3

---

## 📁 Folder Structure

```
CADD_PROJECT_CODE/
├── data/                     # Raw + processed SMILES, labels, and embeddings
├── output/                   # Model results, weights, tokenizer vocab
├── params/                   # Hyperparameter configs (MCA, RNN, etc.)
├── public/trained_models/   # Publicly exported weights for MCA model
├── scripts/                 # Training and evaluation scripts
├── toxsmi/                  # Source code - models and utilities
├── requirements.txt         # Python dependencies
└── setup.py                 # Installation script
```

---

## 🧠 Key Learnings

- **SMILES augmentation** improves generalization significantly.
- **Attention layers** help identify toxic substructures without manual annotations.
- **MCA models** outperform more complex baselines using simple, interpretable components.
- **Uncertainty estimation** (Monte Carlo Dropout + TTA) enhances reliability on unknown molecules.

---

## 🚧 Limitations

- High computational cost (especially for fine-tuned RNNs).
- Limited generalization beyond Tox21 (needs testing on other datasets like SIDER, ClinTox).
- Black-box nature of models reduces explainability for regulatory settings.

---

## 🌱 Future Work

- Add **attention visualization** to highlight toxic substructures.
- Integrate **graph-based models (e.g., GNNs)** for structure-based learning.
- Perform **benchmarking across diverse toxicity datasets**.
- Package as a web API or tool for drug discovery pipelines.

---

## 🤝 Contributors

- **Nikhil Kumar** (2022322)  
- **Aditya Kumar Sinha** (2022034)  
- **Nutan Kumari** (2022341)  
- **Harsh Vishwakarma** (2022205)

---

## 📚 References

- Zhang et al., *Chemical representation learning for toxicity prediction*, Chem. Data. Des., 2023  
- Wu et al., *MoleculeNet: A benchmark for molecular ML*, Chem. Sci., 2018  
- Chen et al., *Graph Contrastive Learning in Chemistry*, J. Chem. Inf. Model., 2022  
- RDKit, Keras, DeepChem libraries

---

> 🔬 Developed as part of the **Computer Aided Drug Design** course project.
