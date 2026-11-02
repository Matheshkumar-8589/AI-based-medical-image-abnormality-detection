"""Flask REST API for the medical image abnormality detector.

Run:  python app.py        (serves http://127.0.0.1:5000)
"""
import os
import time

from flask import Flask, jsonify, request
from flask_cors import CORS

from config import (ALLOWED_EXTENSIONS, MAX_UPLOAD_MB, MODEL_NAME, MODEL_PATH, THRESHOLD)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app)   # lets the separate frontend (another port) call this API


def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/api/health")
def health():
    return jsonify(status="ok", model=MODEL_NAME, model_ready=os.path.exists(MODEL_PATH))


@app.post("/api/predict")
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="No image was uploaded."), 400
    if not allowed(file.filename):
        return jsonify(error="Unsupported file type. Use PNG, JPG or BMP."), 400

    try:
        # Imported here so the server (and /api/health) start even before TensorFlow loads.
        from gradcam import make_gradcam, overlay, to_data_url
        from model import load_trained_model
        from preprocess import preprocess

        model = load_trained_model()
        rgb, x = preprocess(file.read())

        start = time.perf_counter()
        prob = float(model.predict(x[None, ...], verbose=0)[0][0])
        cam = make_gradcam(model, x)
        elapsed_ms = round((time.perf_counter() - start) * 1000)
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503
    except ValueError as e:
        return jsonify(error=str(e)), 422

    abnormal = prob >= THRESHOLD
    return jsonify(
        label="Abnormal" if abnormal else "Normal",
        abnormal_probability=round(prob, 4),
        confidence=round(prob if abnormal else 1 - prob, 4),
        threshold=THRESHOLD,
        model=MODEL_NAME,
        inference_ms=elapsed_ms,
        original=to_data_url(rgb),
        heatmap=to_data_url(overlay(rgb, cam)),
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"File is larger than {MAX_UPLOAD_MB} MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
