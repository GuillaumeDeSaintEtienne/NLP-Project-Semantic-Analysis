# 🧠 NLP-Project-Semantic-Analysis
**Project – Semantic Analysis**

## 📖 Description
This project focuses on performing **semantic analysis** using Natural Language Processing (NLP) techniques.  
It analyzes and compares text data (e.g., job offers and competencies) to extract and measure **semantic similarity** between entities.

The repository includes:
- Input data in CSV format,
- A **Streamlit interface** for visualization and interaction,
- A **Jupyter Notebook** for experimentation,
- A **main Python script** for semantic analysis.

---

## 🗂️ Project Structure

```
NLP-PROJECT-SEMANTIC-ANALYSIS/
│
├── data/
│   ├── competencies.csv        # Competencies dataset
│   └── jobs.csv                # Job offers dataset
│
├── interface/
│   └── front.py                # Front Streamlit web app
│
├── NLP/
│   ├── main.ipynb              # Jupyter notebook for testing
│   └── main.py                 # Main Python script
│
├── .gitignore
├── dockerfile
├── Project - Semantic Analysis.pdf   # Project documentation
├── README.md
└── requirements.txt           # Python dependencies
```

---

## ⚙️ Installation & Setup

### Clone the repository
```bash
git clone https://github.com/GuillaumeDeSaintEtienne/NLP-Project-Semantic-Analysis.git
cd NLP-Project-Semantic-Analysis
```

### Create and activate a virtual environment

On **Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

On **macOS / Linux**:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Install dependencies
Once the virtual environment is activated, install the required Python libraries:
```bash
pip install -r requirements.txt
```
To avoid conflict with notebook
```bash
pip install nbdime
nbdime config-git --enable
```
---

### Run the project

#### ▶️ Run the main script
This script performs the core semantic analysis:
```bash
python NLP/main.py
```

#### 🌐 Launch the Streamlit interface
To visualize and interact with results:
```bash
streamlit run interface/exemple_streamlit.py
```

---

### Explore the Jupyter Notebook
The file `NLP/main.ipynb` contains test cells and visualizations.  
Open it with **Jupyter Notebook** or **VS Code**:
```bash
jupyter notebook NLP/main.ipynb
```

---

### (Optional) Run with Docker
If you prefer to use Docker:

```bash
docker build -t nlp-semantic .
docker run -p 8501:8501 nlp-semantic
```

Once the container is running, open:  
👉 [http://localhost:8501](http://localhost:8501)

---

### Quick command summary
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # or: source venv/bin/activate on Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the main script
python NLP/main.py

# Launch the Streamlit interface
streamlit run interface/exemple_streamlit.py
```

---

## Technologies
- **Python 3.10+**
- **Streamlit** – Web app interface  
- **Pandas / NumPy** – Data processing  
- **scikit-learn / spaCy / transformers** – NLP and embeddings  
- **Jupyter Notebook** – Interactive experimentation  

---

## Goal
To analyze and compare **job descriptions** and **competencies** based on **semantic similarity**, helping match profiles and labor market needs.

---

## Author
Academic project developed for research in **Natural Language Processing (NLP)** and **Semantic Analysis**.

**Developed by :**
- GALLIOU Mael
- GAUDE Corentin
- COUDEVILLE Masao
- DE SAINT ETIENNE Guillaume
- GODET Emilien
- FRANCFORT Sacha
- BOULIC Alexis

**ECE - ING5 Data & IA Inter - Grp 1 - 2025**