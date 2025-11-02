# 🧬 Predicting Transcription Factor Binding Sites Using TBiNet

This repository implements **TBiNet (Transcription Binding Interaction Network)** — a deep learning model for predicting **transcription factor binding sites** from genomic DNA sequences.

---

## 📖 Overview

The model integrates **Convolutional Neural Networks (CNN)**, **Bidirectional LSTMs**, and an **Attention mechanism** to enhance interpretability and predictive performance.  
It is inspired by the work of Park *et al.*, published in **Scientific Reports (2020)**.

---

## 🧩 Reference Papers

**1. Primary Paper:**  
Park, S., Koh, Y., Jeon, H. et al. *Enhancing the interpretability of transcription factor binding site prediction using attention mechanism.*  
**Scientific Reports (2020)**. [https://doi.org/10.1038/s41598-020-77889-2](https://doi.org/10.1038/s41598-020-77889-2)

**2. Related Dataset Paper (DeepSEA):**  
Zhou, J., Troyanskaya, O. G. *Predicting effects of noncoding variants with deep learning–based sequence model.*  
**Nature Methods (2015)**. [https://www.nature.com/articles/s41598-020-70218-4](https://www.nature.com/articles/s41598-020-70218-4)

---

## 🧠 Model Implementation

The training and evaluation are implemented in the Jupyter notebook:  
📘 **`tbinet.ipynb`**

It includes the following stages:

1. **Data Loading** — Uses DeepSEA `.mat` files for training, validation, and testing.  
2. **Model Definition** — Defines CNN, Attention, and BiLSTM layers.  
3. **Training** — Optimized using binary cross-entropy loss and Adam optimizer.  
4. **Evaluation** — Computes AUROC and AUPR metrics to assess model performance.

---

## 📦 Dataset

The dataset can be downloaded from the **DeepSEA** repository:

🔗 [https://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz](https://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz)

After downloading, extract and organize files as follows:

```
data/
├── train.mat
├── valid.mat
└── test.mat
```

---

## ⚙️ Requirements

Make sure you have the following dependencies installed:

```bash
pip install tensorflow keras numpy scipy scikit-learn pandas h5py
```

For macOS (Apple Silicon) users:
```bash
pip install tensorflow-macos tensorflow-metal
```

---

## 🚀 Running the Notebook

1. Open **JupyterLab** or **Jupyter Notebook**.  
2. Launch and execute each cell in `tbinet.ipynb`.  
3. The model will train using DeepSEA data and save checkpoints automatically.

Checkpoints and trained model files are saved to:

```
./checkpoints/
./model/tbinet.keras
```

Example training output:
```
Epoch 24: early stopping
✅ Model saved successfully to: ./model/tbinet.keras
```

---

## 📈 Evaluation

After training, the model computes:

- **Validation Loss:** 0.0528  
- **Average AUROC:** ~0.89  
- **Average AUPR:** ~0.41

These results may vary slightly depending on training sample size and random seed initialization.

---

## 🧬 Citation

If you use this repository or model in your research, please cite:

> Park, S., Koh, Y., Jeon, H. et al. *Enhancing the interpretability of transcription factor binding site prediction using attention mechanism.* Scientific Reports (2020).  
> DOI: [10.1038/s41598-020-77889-2](https://doi.org/10.1038/s41598-020-77889-2)

and the DeepSEA dataset:

> Zhou, J., Troyanskaya, O. G. *Predicting effects of noncoding variants with deep learning–based sequence model.* Nature Methods (2015).  
> [https://www.nature.com/articles/s41598-020-70218-4](https://www.nature.com/articles/s41598-020-70218-4)

---

## 👩‍🔬 Author

**Azami**  
Bioinformatics Researcher  
📧 [Your email or GitHub link here]
