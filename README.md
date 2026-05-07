# 📚 InkLogic — ML-Based Book Recommender System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  <b>Discover your next great read — powered by collaborative filtering on 1.1 million real reader ratings.</b>
</p>

---

## ✦ Overview

**InkLogic** is an end-to-end Machine Learning system that recommends books using **collaborative filtering**. Rather than analysing book content, it identifies readers with similar taste and surfaces books they loved — often uncovering cross-genre gems you'd never find on a bestseller list.

The system is built as a production-grade Streamlit web app, fully containerised with Docker, and follows a modular ML pipeline architecture.

---

## 🗂️ Project Structure

```
InkLogic/
│
├── books_recommender/               # Core Python package
│   ├── components/
│   │   ├── stage_00_data_ingestion.py
│   │   ├── stage_01_data_validation.py
│   │   ├── stage_02_data_transformation.py
│   │   └── stage_03_model_trainer.py
│   ├── config/
│   │   └── configuration.py
│   ├── constant/
│   ├── entity/
│   │   └── config_entity.py
│   ├── exception/
│   │   └── exception_handler.py
│   ├── logger/
│   │   └── log.py
│   ├── pipeline/
│   │   └── training_pipeline.py
│   └── utils/
│       └── util.py
│
├── artifacts/                       # Generated model artifacts (gitignored)
├── config/
│   └── config.yaml                  # Central pipeline configuration
├── notebook/
│   └── research.ipynb               # EDA & prototyping notebook
├── templates/
│   └── book_names.pkl               # Serialised book name list
│
├── app.py                           # Streamlit web application
├── main.py                          # CLI training entry-point
├── setup.py                         # Package setup
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── .gitignore
```

---

## ⚙️ ML Pipeline Workflow

```
config.yaml  →  entity  →  configuration.py  →  components  →  pipeline  →  app.py
```

| Stage | Component | Description |
|-------|-----------|-------------|
| 00 | Data Ingestion | Loads BX-Books, BX-Users, BX-Ratings CSVs |
| 01 | Data Validation | Validates schema, dtypes, and missing values |
| 02 | Data Transformation | Builds user–book rating pivot matrix; filters sparse entries |
| 03 | Model Trainer | Trains KNN collaborative filtering model; serialises artefacts |

---

## 📊 Dataset

**Book-Crossing Dataset** — collected by Cai-Nicolas Ziegler from the Book-Crossing community.

| File | Records | Description |
|------|---------|-------------|
| `BX-Books.csv` | 271,360 | ISBN, title, author, year, publisher, cover URLs |
| `BX-Users.csv` | 278,858 | User ID, location, age |
| `BX-Book-Ratings.csv` | 1,149,780 | Explicit (1–10) and implicit (0) ratings |

> Source: [Kaggle — Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)

---

## 🤖 Algorithm

InkLogic uses **K-Nearest Neighbours (KNN) Collaborative Filtering**:

1. A **user–book rating matrix** is constructed (rows = books, columns = users).
2. Books with fewer than **50 ratings** are filtered to reduce noise.
3. A **KNN model** (k = 5, cosine similarity) is fitted on the matrix.
4. At inference time, the model finds the **5 nearest neighbour books** in rating-space to the query book and returns them as recommendations.

No book content (title, genre, description) is ever read — recommendations emerge purely from collective reader behaviour.

---

## 🚀 How to Run Locally

### Prerequisites

- Python 3.7+
- pip / conda

### Step 1 — Clone the Repository

```bash
git clone https://github.com/beingkabirahmed/InkLogic.git
cd InkLogic
```

### Step 2 — Create a Virtual Environment

```bash
# Using conda
conda create -n inklogic python=3.7.10 -y
conda activate inklogic

# Or using venv
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Add the Dataset

Place the three CSV files in the project root (or configure the path in `config/config.yaml`):

```
BX-Books.csv
BX-Users.csv
BX-Book-Ratings.csv
```

### Step 5 — Train the Model

```bash
python main.py
```

Or use the **Train Model** tab inside the Streamlit app.

### Step 6 — Launch the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🐳 Docker Deployment

### Build & Run Locally

```bash
docker build -t inklogic:latest .
docker run -d -p 8501:8501 inklogic:latest
```

### Deploy to AWS EC2

```bash
# On your EC2 instance (Ubuntu)
sudo apt-get update -y && sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker

# Clone and run
git clone https://github.com/beingkabirahmed/InkLogic.git
cd InkLogic

docker build -t inklogic:latest .
docker run -d -p 8501:8501 inklogic:latest
```

> ⚠️ Make sure port **8501** is open in your EC2 security group inbound rules.

### Push to Docker Hub

```bash
docker login
docker tag inklogic:latest <your-dockerhub-username>/inklogic:latest
docker push <your-dockerhub-username>/inklogic:latest
```

---

## 🖥️ App Features

| Feature | Description |
|---------|-------------|
| 🔍 **Book Search** | Searchable dropdown of all indexed titles |
| ✨ **5 Recommendations** | Book covers + titles from collaborative filtering |
| ⚙️ **In-app Training** | Trigger the full ML pipeline from the UI |
| 📖 **How It Works** | Interactive explanation of the algorithm |
| 📱 **Responsive Layout** | Clean grid that adapts to any screen width |

---

## 📦 Requirements

```
scikit-learn
pandas
numpy
PyYAML
streamlit
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push and open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Kabir Ahmed**  
[GitHub](https://github.com/beingkabirahmed) · [LinkedIn](https://linkedin.com/in/beingkabirahmed)

---

<p align="center">Built with ❤️ using Python, scikit-learn & Streamlit</p>
