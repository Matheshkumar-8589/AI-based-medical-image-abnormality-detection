import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMG_SIZE = (224, 224)
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))        # probability above this = Abnormal
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models", "best_model.keras"))
MODEL_NAME = os.getenv("MODEL_NAME", "DenseNet121")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
MAX_UPLOAD_MB = 10

# Class order used in training: index 0 = normal, index 1 = abnormal
CLASS_NAMES = ["normal", "abnormal"]
