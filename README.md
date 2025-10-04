# 🧠 Handwritten Digit Recognition — Streamlit + FastAPI + Postgres (Dockerized)

Interactive web app for recognizing handwritten digits (0–9) using machine learning models trained on the **MNIST dataset**. The MNIST (Modified National Institute of Standards and Technology) database contains 70,000 images of handwritten digits and is a benchmark dataset for training image classification algorithms.

**Models included:** Multiple trained classifiers (ExtraTrees, Random Forest, SVM, SVM+PCA) that achieve high accuracy on digit recognition tasks.

Frontend in **Streamlit**, backend logic (preprocessing + models) and **PostgreSQL** for storing predictions. All containerized with **Docker Compose**.

---

## 📋 Description

- ✏️ Draw a digit on an interactive canvas
- 🤖 Predict with one of several trained MNIST models (ExtraTrees, Random Forest, SVM, SVM+PCA)
- 📊 See class probabilities and confidence
- 💾 (Optional) Log predictions to Postgres for auditing and analysis
- 🧩 Clean separation of **frontend** and **backend/db utils**
- 🐳 One-command Docker setup
- 🌐 **Live Demo:** [Try the app here](https://nikeolabi-mnist-streamlit-docker-appdocker-streamlit-app-qs9nzd.streamlit.app/)

---

## 📁 Project Structure

```
MNIST_Streamlit_Docker_App/
├── ModelTraining.py           # Main MNIST model training script - originally done in Jupyter Notebook
├── app/                       # Streamlit frontend
│   ├── Docker_Streamlit_app.py
│   ├── requirements.txt
│   └── Dockerfile
├── backend_db_api/            # Backend logic and DB utilities
│   ├── MNIST_model_backend.py
│   ├── db_utils.py
│   ├── requirements.txt
│   └── __init__.py
├── SavedModels/               # Your model zips/pkls + scaler.pkl, pca_scaler.pkl
├── extracted_models/          # Auto-created; holds extracted .pkl files
├── docker-compose.yml         # Orchestration
├── requirements.txt           # Root dependencies
└── README.md
```
---

## 🛠️ Technologies

- **Python 3.10+**
- **Streamlit** (frontend)
- **scikit-learn**, **NumPy**, **Pillow**, **joblib**
- **PostgreSQL** (storage)
- **Docker & Docker Compose** (or run locally)
- **streamlit-drawable-canvas** (drawing)

---

## 🔧 How It Works

1. **Drawing:** User draws a digit on a 280x280 pixel canvas
2. **Preprocessing:** Image is converted to grayscale, centered, and scaled to 28x28 pixels
3. **Normalization:** Pixels are normalized to [0, 1] range
4. **Prediction:** Trained model analyzes the image and outputs probabilities for each digit
5. **Result:** Displays predicted digit with confidence level and probability histogram

---

## 🧠 About the Models

The models are trained on the famous MNIST dataset, which contains 70,000 images of handwritten digits. The models achieves high recognition accuracy through:

- Image preprocessing (centering, normalization)
- Using different ML algorithms
- Hyperparameter optimization

---

## 📦 Installation (Local, no Docker)

1. **Clone the repo**
```sh
git clone <your-repo-url>
cd MNIST_Streamlit_Docker_App
```

2. **Create & activate venv**
```sh
python -m venv .venv
#### Windows:
.\.venv\Scripts\Activate
#### macOS/Linux:
source .venv/bin/activate
```

3. **Install deps**
```sh
pip install -r app/requirements.txt
```

4. **OPTIONAL:**
```sh
# ensure project root is importable (either of the two works)
set PYTHONPATH=%CD%            # Windows CMD
# $env:PYTHONPATH="$PWD"       # PowerShell
# export PYTHONPATH="$PWD"     # macOS/Linux
```

5. **Run Streamlit**
```sh
streamlit run app/Docker_Streamlit_app.py
```

The app will open at **http://localhost:8501**.

---
**Optional: Start Postgres locally and set your DB connection variables if you want to log predictions.**
### 1. Install PostgreSQL
- The app is specifically designed for PostgreSQL
- Download and install from the official site: <https://www.postgresql.org/download/>.
- During installation, set your database username and password (e.g., user: `postgres`, password: `test`).

### 2. Start the PostgreSQL server
- **Windows:** use pgAdmin or start the PostgreSQL service from Services.
- **macOS/Linux:**
  ```sh
  sudo service postgresql start
  # or
  sudo systemctl start postgresql
  ```

### 3. Create the database
---
Create (or use) the default database:
```sql
CREATE DATABASE postgres;
```
---

If you run your app locally, you can set environment variables before starting Streamlit:

**Windows (PowerShell):**
```powershell
$env:POSTGRES_HOST="your_host"
$env:POSTGRES_DB="your_database"
$env:POSTGRES_USER="your_username"
$env:POSTGRES_PASSWORD="your_password"
```

**macOS/Linux:**
```bash
export POSTGRES_HOST=your_host
export POSTGRES_DB=your_database
export POSTGRES_USER=your_username
export POSTGRES_PASSWORD=your_password
```

The default DB environment variables are:
- **POSTGRES_HOST**: `localhost`
- **POSTGRES_DB**: `postgres`
- **POSTGRES_USER**: `postgres`
- **POSTGRES_PASSWORD**: `test`

### 4. Create the table
```sql
CREATE TABLE IF NOT EXISTS user_predictions (
  id SERIAL PRIMARY KEY,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  model_name VARCHAR(64),
  drawn_digit INTEGER,
  predicted_digit INTEGER,
  probability FLOAT,
  probabilities TEXT,
  correct BOOLEAN,
  background_color VARCHAR(16),
  pen_color VARCHAR(16)
);
```

---

## 🐳 Run with Docker (Recommended)

1. **Build & start all services**
```sh
docker compose up --build
```

2. **Open the app**
- Streamlit: http://localhost:8501  
- Postgres: port **5432**
- Or in DockerDesktop

3. **Connect to the DB:**
```sh
docker exec -it mnistStreamlitDatabase psql -U postgres -d postgres
```
and 

4. **Create the table:**

```sh 
CREATE TABLE IF NOT EXISTS user_predictions (
  id SERIAL PRIMARY KEY,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  model_name VARCHAR(64),
  drawn_digit INTEGER,
  predicted_digit INTEGER,
  probability FLOAT,
  probabilities TEXT,
  correct BOOLEAN,
  background_color VARCHAR(16),
  pen_color VARCHAR(16)
);
```

**Default DB creds**:
- DB: `postgres`
- User: `postgres`
- Password: `test`

---

## 🎨 Usage

1. Open the application in your browser
2. Adjust drawing parameters in the sidebar (optional)
3. Draw a digit from 0 to 9 on the canvas
4. View the recognition result and model confidence level
5. Examine the probability histogram for all digits
6. Click "garbage bean" for a new drawing
7. Click Report section to load the report

---

## 🙌 Credits

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) - Modified National Institute of Standards and Technology database
- [Streamlit](https://streamlit.io/)
- [PostgreSQL](https://www.postgresql.org/)
- [Docker](https://www.docker.com/)

---
