# Face Mask Detection API 😷

A production-ready Deep Learning Face Mask Detection system built using **PyTorch**, **FastAPI**, and **Streamlit**.

The project uses **MobileNetV2 Transfer Learning** to classify faces into:

- ✅ WithMask
- ❌ WithoutMask

It includes:

- REST API with FastAPI
- Modern futuristic Streamlit UI
- Face auto-cropping with OpenCV
- Docker support
- Automated API testing
- Real-time webcam prediction

---

# 🚀 Features

## Core Features

- ⚡ FastAPI backend for high-performance inference
- 🧠 MobileNetV2 deep learning model
- 🎨 Futuristic Streamlit frontend
- 📷 Webcam support
- 🧍 Automatic face detection & cropping
- 🐳 Dockerized deployment
- 🧪 Automated API tests
- ☁️ Ready for cloud deployment

---

# 🛠️ Tech Stack

| Technology | Usage |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| FastAPI | Backend REST API |
| Streamlit | Frontend web interface |
| OpenCV | Face detection |
| MobileNetV2 | Transfer learning model |
| Docker | Containerization |
| NumPy | Numerical processing |
| Pillow | Image handling |

---

# 📂 Project Structure

```bash
.
├── app.py                      # FastAPI backend
├── streamlit_app.py            # Streamlit frontend
├── mask_detector.pth           # Trained model weights
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── start.sh                    # Linux/Mac startup script
├── run.bat                     # Windows startup script
├── test_api.py                 # Automated API testing
├── Advanced_ML_Project.ipynb   # Training notebook
└── README.md
```

---

# 🧠 Model Architecture

This project uses:

## MobileNetV2

A lightweight CNN architecture optimized for:

- Fast inference
- Low latency
- Edge devices
- Real-time computer vision tasks

### Why MobileNetV2?

- High accuracy
- Lightweight
- Fast inference speed
- Perfect for deployment APIs

---

# 🖼️ Image Processing Pipeline

The uploaded image passes through several preprocessing steps:

1. Face Detection using OpenCV Haar Cascade
2. Face Cropping
3. CLAHE enhancement
4. Resize to 224×224
5. Tensor conversion
6. Normalization
7. Prediction using MobileNetV2

---

# 📦 Installation

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/face-mask-detection.git
cd face-mask-detection
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

# Run FastAPI Backend

```bash
python app.py
```

Server will run on:

```bash
http://localhost:8000
```

Swagger Docs:

```bash
http://localhost:8000/docs
```

---

# Run Streamlit Frontend

Open another terminal and run:

```bash
streamlit run streamlit_app.py
```

Frontend will run on:

```bash
http://localhost:8501
```

---

# 🐳 Docker Deployment

## Build Docker Image

```bash
docker build -t mask-ai .
```

## Run Container

```bash
docker run -p 8000:8000 -p 8501:8501 mask-ai
```

---

# 📡 API Endpoints

# GET /

Returns API status.

### Example

```json
{
  "message": "Face Mask Detection API is running",
  "docs": "/docs"
}
```

---

# GET /health

Returns server health information.

### Example

```json
{
  "status": "ok",
  "device": "cpu"
}
```

---

# POST /predict

Upload image for mask detection.

## Request

Multipart form-data:

```text
file=image.jpg
```

## Response Example

```json
{
  "status": "mask_on",
  "action": "Allow entry",
  "class": "WithMask",
  "confidence": 0.9821,
  "probabilities": {
    "WithMask": 0.9821,
    "WithoutMask": 0.0179
  }
}
```

---

# 🎨 Streamlit UI Features

The frontend includes:

- Futuristic cyberpunk design
- Neon animations
- Real-time confidence bars
- Live webcam capture
- Image upload support
- Auto face crop toggle
- Status monitoring

---

# 🧪 Automated Testing

Run the API test suite:

```bash
python test_api.py
```

Tests include:

- API health check
- Valid image prediction
- Invalid file handling
- Empty request handling
- PNG/JPG compatibility
- Large image handling
- Performance benchmark

---

# ⚙️ Requirements

Main dependencies:

```txt
fastapi
uvicorn
torch
torchvision
opencv-python-headless
streamlit
numpy
Pillow
python-multipart
requests
```

---

# 📈 Performance

## Optimizations Used

- MobileNetV2 lightweight architecture
- CLAHE preprocessing
- Torch no_grad inference
- Efficient image resizing
- Face cropping before inference

---

# 🔐 Possible Future Improvements

- JWT Authentication
- Multi-face detection
- Video stream processing
- GPU acceleration
- Database logging
- User dashboard
- Cloud deployment
- Kubernetes support

---

# ☁️ Deployment Options

This project can be deployed on:

- Render
- Railway
- AWS EC2
- Azure
- Google Cloud
- Docker VPS
- Hugging Face Spaces

---

# 🧑‍💻 Developer

Developed by Mohamed Ahmed

AI & Data Science Student

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ Support

If you like this project:

- Star the repository
- Fork the project
- Share it with others

---

# 📬 Contact

GitHub:
https://github.com/YOUR_USERNAME

LinkedIn:
https://linkedin.com/in/YOUR_PROFILE
