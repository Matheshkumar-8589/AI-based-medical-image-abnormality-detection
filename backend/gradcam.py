import base64

import cv2
import numpy as np
import tensorflow as tf


def _last_conv_layer(model):
    for layer in reversed(model.layers):
        shape = getattr(layer, "output", None)
        if shape is not None and len(layer.output.shape) == 4:
            return layer.name
    raise ValueError("No convolutional layer found for Grad-CAM.")


def make_gradcam(model, x: np.ndarray) -> np.ndarray:
    """x: (224, 224, 3) float32. Returns a 0-1 heat-map of the abnormal class."""
    grad_model = tf.keras.Model(
        model.input, [model.get_layer(_last_conv_layer(model)).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(x[None, ...])
        score = pred[:, 0]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, heat, alpha, 0)


def to_data_url(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
