"""Train the classifier.

Dataset layout (folder names are the class names):
    dataset/train/normal   dataset/train/abnormal
    dataset/val/normal     dataset/val/abnormal
    dataset/test/normal    dataset/test/abnormal

Run:  python train.py
"""
import os

import cv2
import matplotlib
import numpy as np
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight

from config import BASE_DIR, CLASS_NAMES, IMG_SIZE, MODEL_PATH
from model import build_model, compile_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = os.path.join(BASE_DIR, "dataset")
OUT = os.path.join(BASE_DIR, "outputs")
BATCH = 32
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def _clahe_batch(batch):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = [clahe.apply(im[..., 0].astype("uint8")) for im in batch]
    return np.stack(out)[..., None].astype("float32")


def prep(x, y):                       # same steps as preprocess.py
    x = tf.numpy_function(_clahe_batch, [x], tf.float32)
    x.set_shape([None, IMG_SIZE[0], IMG_SIZE[1], 1])
    return tf.image.grayscale_to_rgb(x) / 255.0, y


augment = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomBrightness(0.1, value_range=(0, 1)),
])


def load(split, training=False):
    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA, split), labels="inferred", label_mode="binary",
        class_names=CLASS_NAMES, color_mode="grayscale", image_size=IMG_SIZE,
        batch_size=BATCH, shuffle=training, seed=42)
    ds = ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


train_ds, val_ds, test_ds = load("train", True), load("val"), load("test")

labels = np.concatenate([y.numpy().ravel() for _, y in load("train")])
cw = compute_class_weight("balanced", classes=np.array([0.0, 1.0]), y=labels)
class_weights = {0: cw[0], 1: cw[1]}

model, base = build_model("densenet121")
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
]

# Phase 1: train the new head with the backbone frozen
compile_model(model, 1e-4)
h1 = model.fit(train_ds, validation_data=val_ds, epochs=10,
               class_weight=class_weights, callbacks=callbacks)

# Phase 2: fine-tune the top of the backbone with a small learning rate
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
compile_model(model, 1e-5)
h2 = model.fit(train_ds, validation_data=val_ds, epochs=20,
               class_weight=class_weights, callbacks=callbacks)

# Training curves
hist = {k: h1.history[k] + h2.history[k] for k in h1.history}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for a, m in zip(ax, ("loss", "accuracy")):
    a.plot(hist[m], label="Train"); a.plot(hist["val_" + m], label="Validation")
    a.set_title(m.capitalize()); a.set_xlabel("Epoch"); a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "training_curves.png"), dpi=150); plt.close()

# Test-set evaluation
best = tf.keras.models.load_model(MODEL_PATH)
y_true = np.concatenate([y.numpy().ravel() for _, y in test_ds]).astype(int)
y_prob = best.predict(test_ds, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
print("AUC-ROC:", round(roc_auc_score(y_true, y_prob), 4))
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=CLASS_NAMES, cmap="Blues")
plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=150)
print("Saved model to", MODEL_PATH)
