"""Image preprocessing shared by training and inference (keep them identical)."""
import cv2
import numpy as np

from config import IMG_SIZE

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def decode_gray(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read the file as an image. Upload a PNG or JPG scan.")
    return img


def preprocess(image_bytes: bytes):
    """Returns (display_rgb_uint8, model_input_float32 of shape (224, 224, 3))."""
    gray = decode_gray(image_bytes)
    gray = cv2.resize(gray, IMG_SIZE)
    gray = _clahe.apply(gray)                        # contrast enhancement
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)     # 3 channels for ImageNet backbones
    return rgb, rgb.astype("float32") / 255.0        # scale to 0-1
