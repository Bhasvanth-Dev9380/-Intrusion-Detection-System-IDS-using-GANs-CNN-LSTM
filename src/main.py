# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ✅ Step 1: Data Preprocessing
# Load dataset
dataset_path = "../datasets/NLS_KDD_Original.csv"
data = pd.read_csv(dataset_path)

# Handle missing values
data.fillna(0, inplace=True)

# ✅ Select Numerical Features Automatically
numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
numerical_columns.remove('Class')  # Exclude target variable

# Normalize numerical features
scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# ✅ One-hot encode categorical features
categorical_columns = ['Protocol Type', 'Flag', 'Service']
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_categorical = encoder.fit_transform(data[categorical_columns]).toarray()
encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_columns))

# Merge encoded data and drop original categorical columns
data = pd.concat([data, encoded_categorical_df], axis=1).drop(columns=categorical_columns)

# ✅ Split dataset into features and labels
X_real = data.drop(columns=['Class'])  # Features
y_real = data['Class']  # Target (attack type or normal)

# Convert labels to one-hot encoding
num_classes = len(np.unique(y_real))
y_real_onehot = tf.keras.utils.to_categorical(y_real, num_classes=num_classes)

# ✅ Load Trained GAN Model (Synthetic Intrusion Generation)
generator = tf.keras.models.load_model("../models/generator.keras")
latent_dim = 100
synthetic_noise = np.random.normal(0, 1, (len(X_real) // 2, latent_dim))  # Generate synthetic samples
X_synthetic = generator.predict(synthetic_noise)

# ✅ Ensure Synthetic Data Matches Real Data Shape
if X_synthetic.shape[1] < X_real.shape[1]:
    print(f"⚠️ Warning: Padding X_synthetic from {X_synthetic.shape[1]} to {X_real.shape[1]}")
    X_synthetic = np.pad(X_synthetic, ((0, 0), (0, X_real.shape[1] - X_synthetic.shape[1])), 'constant')

elif X_synthetic.shape[1] > X_real.shape[1]:
    print(f"⚠️ Warning: Trimming X_synthetic from {X_synthetic.shape[1]} to {X_real.shape[1]}")
    X_synthetic = X_synthetic[:, :X_real.shape[1]]

# ✅ Create attack labels for synthetic data (Assuming attack class index is last)
y_synthetic = np.zeros((len(X_synthetic), num_classes))
y_synthetic[:, -1] = 1  # Assign synthetic samples to the last attack class

# ✅ Merge real & synthetic data
X_combined = np.vstack((X_real, X_synthetic))
y_combined = np.vstack((y_real_onehot, y_synthetic))


# Shuffle dataset
shuffle_indices = np.random.permutation(len(X_combined))
X_combined, y_combined = X_combined[shuffle_indices], y_combined[shuffle_indices]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# ✅ Step 2: Autoencoder for Feature Extraction
input_dim = X_train.shape[1]
encoding_dim = 32  # Latent space dimension

input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = models.Model(input_layer, decoded)
encoder = models.Model(input_layer, encoded)

# Compile and train autoencoder
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=10, batch_size=128, validation_data=(X_test, X_test))

# ✅ Extract Features Using the Encoder
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# ✅ Step 3: CNN-LSTM for Classification
from tensorflow.keras.layers import Conv1D, LSTM, Dense

# Reshape data for CNN-LSTM
X_train_reshaped = X_train_encoded.reshape(-1, X_train_encoded.shape[1], 1)
X_test_reshaped = X_test_encoded.reshape(-1, X_test_encoded.shape[1], 1)

# Define CNN-LSTM Model
cnn_lstm = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)),
    LSTM(32, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')  # Multiclass classification
])
cnn_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Cross-Validation Training
kf = KFold(n_splits=3)
fold = 1

for train_idx, val_idx in kf.split(X_train_reshaped):
    print(f"\n🚀 Training Fold {fold}")
    fold += 1
    X_train_fold, X_val_fold = X_train_reshaped[train_idx], X_train_reshaped[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    cnn_lstm.fit(
        X_train_fold, y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        epochs=5,
        batch_size=64
    )

# ✅ Save Trained CNN-LSTM Model
model_save_path = "../models/cnn_lstm_model.h5"
cnn_lstm.save(model_save_path)
print(f"\n✅ Model saved at: {model_save_path}")

# ✅ Step 4: Evaluate the System
y_pred_prob = cnn_lstm.predict(X_test_reshaped)  # Predicted probabilities
y_pred = np.argmax(y_pred_prob, axis=1)  # Predicted class indices
y_test_class = np.argmax(y_test, axis=1)  # True class indices

# Compute Evaluation Metrics
precision = precision_score(y_test_class, y_pred, average='weighted')
recall = recall_score(y_test_class, y_pred, average='weighted')
f1 = f1_score(y_test_class, y_pred, average='weighted')
accuracy = accuracy_score(y_test_class, y_pred)

# ✅ Fix ROC AUC Calculation
try:
    roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
except ValueError:
    roc_auc = "N/A (ROC AUC cannot be calculated for single-class predictions)"

print("\n📊 Evaluation Metrics:")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1 Score: {f1:.4f}")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ ROC AUC: {roc_auc}")

# ✅ Plot Confusion Matrix
cm = confusion_matrix(y_test_class, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("../models/confusion_matrix.png")  # ✅ Save the confusion matrix instead of displaying
print("\n✅ Confusion Matrix saved at: ../models/confusion_matrix.png")

