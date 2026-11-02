import os

import tensorflow as tf
from tensorflow.keras import applications, layers, models

from config import IMG_SIZE, MODEL_PATH


def build_model(backbone: str = "densenet121"):
    shape = IMG_SIZE + (3,)
    if backbone == "resnet50":
        base = applications.ResNet50(weights="imagenet", include_top=False, input_shape=shape)
    else:
        base = applications.DenseNet121(weights="imagenet", include_top=False, input_shape=shape)
    base.trainable = False                                  # phase 1: train the head only

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(base.input, out), base


def compile_model(model, lr=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )


_model = None


def load_trained_model():
    """Loads the saved model once and caches it."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. Run `python train.py` first."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model
