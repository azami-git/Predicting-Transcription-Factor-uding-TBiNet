# 🧬 Predicting Transcription Factor Binding Sites Using TBiNet

This project implements **TBiNet** (Transcription Binding Interaction Network), a deep learning model designed to **predict transcription factor binding sites** from genomic DNA sequences.  
The model integrates **Convolutional Neural Networks (CNNs)**, **Bidirectional LSTMs**, and an **Attention mechanism** to enhance interpretability and prediction accuracy.

---

## 📄 Reference Paper
**Park, S., Koh, Y., Jeon, H. et al.**  
*Enhancing the interpretability of transcription factor binding site prediction using attention mechanism.*  
**Scientific Reports (2020)**  
[https://doi.org/10.1038/s41598-020-77889-2](https://doi.org/10.1038/s41598-020-77889-2)

---

## 🚀 Features
- 🧠 Deep learning model combining CNN + BiLSTM + Attention.  
- 📈 Predicts transcription factor binding site probabilities from DNA sequences.  
- 💡 Improves interpretability using attention visualization.  
- 💾 Compatible with DeepSEA’s public training dataset format (`.mat` files).  
- 🔁 Supports checkpoint saving and model resuming.

---

## 📦 Dataset
You can download the dataset from **DeepSEA (Princeton University)**:

🔗 [https://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz](https://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz)

After downloading, extract the bundle and locate:
```
train.mat
valid.mat
test.mat
```

Then, place them in the following directory:
```
data/
├── train.mat
├── valid.mat
└── test.mat
```

---

## 💻 Training Script
Training is done using the notebook:
```
tbinet.ipynb
```

It trains the model on `train.mat` and validates on `valid.mat`.  
The training process automatically saves checkpoints to the `checkpoints/` folder and the final trained model to `model/tbinet.keras`.

Example:
```
Epoch 21/60
100/100 ━━━━━━━━━━━━━━━━━━━━ 70s 684ms/step - loss: 0.0379 - val_loss: 0.0505
Epoch 21: early stopping
✅ Model saved to ./model/tbinet.keras
```

---

## 🧠 Model Architecture Overview
1. **Input:** DNA sequence (1000 bp, one-hot encoded).  
2. **Conv1D Layer:** Detects DNA sequence motifs.  
3. **MaxPooling:** Reduces feature dimensionality.  
4. **Attention Layer:** Highlights critical sequence positions.  
5. **BiLSTM:** Captures long-range dependencies.  
6. **Fully Connected Layer:** Integrates learned representations.  
7. **Sigmoid Output:** Predicts transcription factor binding probability.

---

## 🧩 Example Output
After training, you can evaluate the model on `test.mat` to obtain performance metrics such as AUROC and AUPR.

Example evaluation:
```
✅ Validation Loss: 0.0528
Averaged AUROC: 0.89
Averaged AUPR: 0.41
```

---

## ⚙️ Requirements
See `requirements.txt` for full dependencies.

Key packages:
- `tensorflow`
- `keras`
- `numpy`
- `scipy`
- `scikit-learn`
- `pandas`
- `h5py`

For macOS (Apple Silicon):
```bash
pip install tensorflow-macos tensorflow-metal
```

---

## 🧬 Citation
If you use this repository or model in your research, please cite:

> Park, S., Koh, Y., Jeon, H. et al. *Enhancing the interpretability of transcription factor binding site prediction using attention mechanism.* Scientific Reports (2020).

---

## 👩‍🔬 Author
**Azami**  
🔬 Bioinformatics Researcher  
📧 [Your email or GitHub link here]
