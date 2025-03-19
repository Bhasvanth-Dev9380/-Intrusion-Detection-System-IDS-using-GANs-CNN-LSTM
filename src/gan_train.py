import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout
import os

# Create models directory if not exists
os.makedirs("../models", exist_ok=True)

latent_dim = 100  # Noise input size

# ✅ Step 1: Load Dataset
dataset_path = "../datasets/NLS_KDD_Original.csv"
data = pd.read_csv(dataset_path)

# ✅ Step 2: Feature Selection (Remove Non-Numeric Columns)
numeric_features = data.select_dtypes(include=[np.number]).drop(columns=['Class'])  # Drop target column
feature_dim = numeric_features.shape[1]  # Dynamically set feature count

# ✅ Step 3: Normalize Features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_real = scaler.fit_transform(numeric_features)  # Normalize real data

# ✅ Step 4: Define Generator Model
def build_generator():
    model = Sequential([
        Dense(128, input_shape=(latent_dim,)),
        LeakyReLU(negative_slope=0.2),
        BatchNormalization(momentum=0.8),
        Dense(256),
        LeakyReLU(negative_slope=0.2),
        BatchNormalization(momentum=0.8),
        Dense(512),
        LeakyReLU(negative_slope=0.2),
        Dense(feature_dim, activation='tanh')
    ])
    return model

# ✅ Step 5: Define Discriminator Model
def build_discriminator():
    model = Sequential([
        Dense(512, input_shape=(feature_dim,)),
        LeakyReLU(negative_slope=0.2),
        Dropout(0.3),
        Dense(256),
        LeakyReLU(negative_slope=0.2),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

# ✅ Step 6: Initialize Models
generator = build_generator()
discriminator = build_discriminator()

# ✅ Optimizers for Generator & Discriminator
gen_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
disc_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

discriminator.compile(optimizer=disc_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# ✅ Step 7: Create GAN
discriminator.trainable = False
gan = Sequential([generator, discriminator])
gan.compile(optimizer=gen_optimizer, loss='binary_crossentropy')

# ✅ Step 8: Prepare Training Data (Minimal Subset for Quick Test)
batch_size = 32
epochs = 10  # ✅ Reduced to 10 epochs
save_interval = 5  # ✅ Save models every 5 epochs

# ✅ Limit Dataset to 500 Samples for Fast Debugging
real_samples_dataset = tf.data.Dataset.from_tensor_slices(X_real).shuffle(500).batch(batch_size).take(500).prefetch(tf.data.AUTOTUNE)

# ✅ Step 9: Training Function (Debugging Mode)
def train_step(real_samples):
    current_batch_size = tf.shape(real_samples)[0]
    real_labels = tf.ones((current_batch_size, 1))
    fake_labels = tf.zeros((current_batch_size, 1))

    noise = tf.random.normal([current_batch_size, latent_dim])
    fake_samples = generator(noise, training=True)

    # Train Discriminator
    discriminator.trainable = True
    with tf.GradientTape() as tape:
        real_predictions = discriminator(real_samples, training=True)
        fake_predictions = discriminator(fake_samples, training=True)

        real_loss = tf.keras.losses.binary_crossentropy(real_labels, real_predictions)
        fake_loss = tf.keras.losses.binary_crossentropy(fake_labels, fake_predictions)
        d_loss = tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)

    gradients = tape.gradient(d_loss, discriminator.trainable_variables)
    if gradients:
        disc_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))

    discriminator.trainable = False

    # Train Generator
    with tf.GradientTape() as tape:
        fake_samples = generator(noise, training=True)
        fake_predictions = discriminator(fake_samples, training=True)
        g_loss = tf.keras.losses.binary_crossentropy(real_labels, fake_predictions)
        g_loss = tf.reduce_mean(g_loss)

    gradients = tape.gradient(g_loss, generator.trainable_variables)
    if gradients:
        gen_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

    # ✅ Print Loss Every Step for Debugging
    tf.print("Epoch:", epoch + 1, "D Loss:", d_loss, "G Loss:", g_loss)

    return d_loss.numpy().item(), g_loss.numpy().item()

# ✅ Step 10: Training Loop (Ultra-Fast)
for epoch in range(epochs):
    print(f"\n🚀 Starting Epoch {epoch + 1}/{epochs}...")

    for real_samples in real_samples_dataset:
        d_loss, g_loss = train_step(real_samples)

    # ✅ Save Model Every 5 Epochs
    if epoch % save_interval == 0 or epoch == epochs - 1:
        generator.save('../models/generator.keras')
        discriminator.save('../models/discriminator.keras')
        print(f"\n✅ Model Saved at Epoch {epoch}: D Loss = {d_loss:.4f}, G Loss = {g_loss:.4f}")

print("\n✅ GAN Training Completed. Models Saved!")
