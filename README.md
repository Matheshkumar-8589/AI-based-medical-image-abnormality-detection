# AI-Based Medical Image Abnormality Detection System

```
medical-abnormality-detection/
├── backend/            Flask API + model code (Python)
│   ├── app.py          REST API  (/api/health, /api/predict)
│   ├── config.py       settings (image size, threshold, model path)
│   ├── preprocess.py   resize + CLAHE + normalisation
│   ├── model.py        DenseNet121 / ResNet50 transfer-learning model
│   ├── gradcam.py      Grad-CAM heat-map
│   ├── train.py        training + evaluation script
│   ├── dataset/        put your images here (see dataset/README.txt)
│   ├── models/         trained model is saved here (best_model.keras)
│   └── requirements.txt
└── frontend/           Web page (HTML / CSS / JavaScript, no build step)
    ├── index.html
    ├── style.css
    ├── script.js
    └── config.js       API address
```

## 1. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt

# arrange images as described in dataset/README.txt, then train:
python train.py                # saves models/best_model.keras and plots in outputs/

python app.py                  # API runs at http://127.0.0.1:5000
```

## 2. Frontend

In a second terminal:

```bash
cd frontend
python -m http.server 5500
```

Open http://127.0.0.1:5500 in the browser, upload a scan and select **Analyse image**.
If the backend runs on another address, edit `frontend/config.js`.

## API

`POST /api/predict` - multipart form field `image`
Returns: `label`, `abnormal_probability`, `confidence`, `threshold`, `model`,
`inference_ms`, `original` and `heatmap` (PNG data URLs).

`GET /api/health` - `{status, model, model_ready}`

> Decision-support tool only. A qualified radiologist must confirm every finding.
