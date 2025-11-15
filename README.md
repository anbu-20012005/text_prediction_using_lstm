# ✨ Predictive Text Generation using LSTM (TensorFlow + Streamlit)

This project builds a **predictive text generation model** using an LSTM neural network trained on a custom dataset.  
The final model is deployed as a **Streamlit web app**, allowing users to generate text continuations interactively.

🚀 **Live App:**  
👉 https://predictive-text-generation.streamlit.app/

---

## 📘 Features

- ✔️ Custom text dataset preprocessing  
- ✔️ Tokenization & sequence generation  
- ✔️ LSTM-based predictive text model  
- ✔️ TensorFlow deep learning pipeline  
- ✔️ Streamlit UI for real-time prediction  
- ✔️ Fully deployed and accessible online  

---

## 🧠 Model Architecture

The predictive model uses:

- **Embedding Layer**  
- **Stacked LSTM (150 units × 2 layers)**  
- **Dense layer with softmax** (predicts next word)  
- **Sparse categorical crossentropy loss**  

This architecture learns long-term dependencies between words.

---

## 📂 Folder Structure

```

AI_FOR_PREDICTIVE_TEXT/
│
├── app.py                 # Streamlit UI
├── main.py                # Runs preprocessing + training
├── requirements.txt       # Dependencies
│
├── data/
│   └── dataset.txt        # Training dataset
│
├── model/
│   ├── text_predictor.h5  # Trained LSTM model
│   └── tokenizer.json     # Word tokenizer
│
└── training/
├── preprocess.py      # Dataset cleaning + sequence creation
└── train.py           # Model training script



## ⚙️ How It Works

### **1️⃣ Preprocessing**
- Loads dataset  
- Tokenizes text  
- Converts lines into input sequences  
- Pads sequences  
- Creates integer labels  
- Saves tokenizer for later use  

### **2️⃣ Training**
- Builds a stacked LSTM model  
- Trains on input sequences  
- Saves final `.h5` model  

### **3️⃣ Streamlit App**
- Loads the saved model + tokenizer  
- User enters a starting phrase  
- App predicts the next 10+ words  
- Displays generated text in real time  



## 🧪 Local Usage

### Clone the repo:
```bash
git clone https://github.com/anbu-20012005/text_prediction_using_lstm.git
````

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Run Streamlit app locally:

```bash
streamlit run app.py
```

---

## 🌐 Deployment

This project is deployed using **Streamlit Community Cloud**.

The app is accessible here:

👉 **[https://predictive-text-generation.streamlit.app/](https://predictive-text-generation.streamlit.app/)**


## 🙌 Acknowledgements

* **TensorFlow** — deep learning framework
* **Keras Tokenizer** — text vectorization
* **Streamlit** — fast UI deployment
* Dataset sourced from uploaded text file



## 📧 Contact

If you'd like to improve this project or collaborate, feel free to open an issue or PR.

