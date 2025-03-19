# 🔥 Intrusion Detection System (IDS) using GANs & CNN-LSTM

## 🚀 Overview
This project implements an **Intrusion Detection System (IDS)** using:
- **GANs** to generate synthetic attack data.
- **CNN-LSTM** model to classify network traffic.
- **Streamlit** for real-time IDS simulation.

## 📌 Features
✔ **Generates synthetic intrusion data using GANs**  
✔ **CNN-LSTM for intrusion detection**  
✔ **Live simulation of incoming network packets**  
✔ **Confidence-based attack detection**  
✔ **Interactive UI with Streamlit**  

---

## 🛠️ Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

---

## 🔥 Running the IDS Pipeline
### Step 1️⃣ Train the GAN Model
```sh
python src/gan_train.py
```
✅ This will train a **GAN model** to generate synthetic attack data.

### Step 2️⃣ Train the CNN-LSTM Model
```sh
python src/main.py
```
✅ This will train a **CNN-LSTM model** for intrusion detection using both real & synthetic data.

### Step 3️⃣ Run the IDS Simulation
```sh
streamlit run deployment/app.py
```
✅ This will start the **real-time intrusion detection** UI.

---

## 📊 Live IDS Simulation
- Click **Start Live Simulation** in the Streamlit app.
- View **incoming packets** and **intrusion detection logs** in real-time.
- Packets with **low confidence (< 70%)** are marked as **attacks**.
- **Confusion matrix** visualization is displayed at the end.

---

## 📁 Project Structure
```
📂 intrusion
│── 📂 datasets                 # Dataset folder
│   ├── NLS_KDD_Original.csv
│
│── 📂 deployment               # Streamlit IDS simulation
│   ├── app.py                  # Streamlit UI
│
│── 📂 models                   # Trained models & results
│   ├── cnn_lstm_model.h5
│   ├── confusion_matrix.png
│   ├── discriminator.keras
│   ├── generator.keras
│
│── 📂 src                      # Source code
│   ├── gan_train.py            # Train GAN model
│   ├── main.py                 # Train CNN-LSTM
│
│── requirements.txt            # Python dependencies
│── readme.md                   # Project documentation
```

---

## 🏆 Results & Performance
✅ **Precision:** `99%+`  
✅ **Recall:** `99%+`  
✅ **F1 Score:** `99%+`  
✅ **Accuracy:** `99%+`  
✅ **ROC AUC Score:** `99%+`

---

## 💡 Future Enhancements
- ✅ **Integrate with real-time packet capture**
- ✅ **Deploy the IDS as a cloud-based service**
- ✅ **Optimize GAN & CNN-LSTM for better performance**

---



