# Machine Failure Prediction API

A REST API built with **FastAPI** that predicts machine failure using a trained Random Forest model.

- **ROC-AUC Score: 0.92**
- **Accuracy: 84%**

---

## 📁 Files in this folder

| File | Description |
|------|-------------|
| `main.py` | FastAPI application |
| `machine_failure_model.pkl` | Trained ML model |
| `requirements.txt` | Python dependencies |
| `render.yaml` | Config for Render deployment |
| `Procfile` | Config for Railway deployment |

---

## 🚀 Deploy for FREE on Render.com (Recommended)

1. **Create a GitHub repo** and push all files in this folder to it
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. Go to **[render.com](https://render.com)** → Sign up (free) → **New → Web Service**

3. Connect your GitHub repo

4. Use these settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment:** Python 3

5. Click **Deploy** — your live API URL will be:
   ```
   https://machine-failure-api.onrender.com
   ```

---

## 🚀 Alternative: Deploy on Railway.app

1. Go to **[railway.app](https://railway.app)** → New Project → Deploy from GitHub
2. Connect your repo — Railway auto-detects the `Procfile`
3. Done! Live in ~2 minutes.

---

## 🧪 Test the API locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open your browser at: **http://localhost:8000/docs** (interactive Swagger UI)

---

## 📡 API Endpoints

### `POST /predict`
Predict failure for a single machine reading.

**Request body:**
```json
{
  "time_cycle_count": 6151.12,
  "temperature_c": 74.13,
  "pressure_bar": 165.51,
  "vibration_mm_s": 2.21,
  "speed_rpm": 1489.87,
  "torque_nm": 208.19,
  "operational_mode": "Normal"
}
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Machine Failure",
  "failure_probability": 0.82,
  "safe_probability": 0.18,
  "confidence": "High"
}
```

### `POST /predict/batch`
Send up to 500 records at once (same format as above, wrapped in a list `[...]`).

### `GET /health`
Health check endpoint — returns `{"status": "healthy"}`.

---

## 📌 Operational Mode values
The `operational_mode` field accepts one of:
- `Normal`
- `Idle`
- `Overload`
- `Maintenance`

---

## 📊 Model Info
- **Algorithm:** Random Forest (300 trees)
- **Preprocessing:** Median imputation + StandardScaler for numerics, OneHotEncoding for categorical
- **Training data:** 8,000 rows | **Test data:** 2,000 rows
- **Handles missing values:** Yes (automatically imputed)
