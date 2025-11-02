# 🧬 TBiNet — Deep Learning Model for Genomic Sequence Analysis

TBiNet (Transcription Binding Interaction Network) is a deep learning model that predicts genomic regulatory features from DNA sequences using a **Convolutional Neural Network (CNN)**, **Bi-directional LSTM**, and **Attention mechanism**.

This architecture is inspired by [DeepSEA](https://www.nature.com/articles/nmeth.3547) and optimized for training on genomic datasets in `.mat` format.

---

## 🚀 Features
- ✅ 1D Convolution + MaxPooling for motif detection  
- ✅ Attention mechanism for positional weighting  
- ✅ BiLSTM layer for long-range sequence dependencies  
- ✅ Fully Connected layer for feature integration  
- ✅ Compatible with `.mat` datasets (train/valid/test)  
- ✅ Keras/TensorFlow implementation with checkpoint saving  

---

## 🧱 Project Structure
```
TBiNet/
│
├── notebooks/
│   ├── train.ipynb        # training script
│   ├── test.ipynb         # evaluation notebook
│
├── data/
│   ├── train.mat          # training data
│   ├── valid.mat          # validation data
│   ├── test.mat           # test data
│
├── checkpoints/           # model checkpoints (auto-generated)
│   ├── tbinet.01-0.05.keras
│   ├── ...
│
├── model/
│   ├── tbinet.keras       # final trained model
│
├── requirements.txt       # dependencies
├── .gitignore             # ignored files
└── README.md              # project documentation
```

---

## ⚙️ Installation

### 1️⃣ Clone this repository
```bash
git clone https://github.com/<your-username>/TBiNet.git
cd TBiNet
```

### 2️⃣ Create a virtual environment
```bash
python3 -m venv env
source env/bin/activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🧠 Training the Model
Run the training notebook:
```bash
jupyter lab
```
Then open:
```
notebooks/train.ipynb
```

Training automatically saves checkpoints to:
```
./checkpoints/
```

And the final model to:
```
./model/tbinet.keras
```

---

## 🧪 Evaluating the Model
After training, evaluate performance using:
```
notebooks/test.ipynb
```

You can compute metrics like **AUROC** and **AUPR** across genomic tasks.

---

## 📈 Example Output
During training:
```
Epoch 21/60
100/100 ━━━━━━━━━━━━━━━━━━━━ 70s 684ms/step - loss: 0.0379 - val_loss: 0.0505
Epoch 21: early stopping
```

Validation Results:
```
✅ Validation Loss: 0.0528
```

---

## 📦 Requirements
See [`requirements.txt`](requirements.txt)

Main dependencies:
- `tensorflow` (with `tensorflow-metal` for macOS)
- `keras`
- `numpy`
- `scipy`
- `scikit-learn`
- `pandas`
- `h5py`
- `matplotlib`

---

## 🧰 Notes for macOS (Apple Silicon)
To enable GPU acceleration:
```bash
pip install tensorflow-macos tensorflow-metal
```

If you encounter Theano errors, you can safely remove any `theano` imports — TensorFlow handles GPU usage directly on macOS.

---

## 📜 License
This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## 👩‍🔬 Author
**Azami**  
🔬 Bioinformatics Researcher  
📧 [Your email or GitHub link here]
