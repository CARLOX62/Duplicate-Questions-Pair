# 🧠 Question Similarity Detection using Advanced BOW Features

This project implements a **machine learning model** to detect whether two questions are duplicates or similar using **advanced Bag of Words (BOW)** features. It includes a **Flask web application** that allows users to input two questions and get real-time predictions on their similarity.

---

## 🚀 Features
- **Text Preprocessing** – Cleans and normalizes text (expands contractions, removes HTML tags, special symbols, etc.)
- **Advanced Feature Engineering**
  - Length-based features (character/word counts, differences)
  - Token-based features (common/unique words, stop words ratio)
  - Fuzzy matching (QRatio, Partial Ratio, Token Sort/Set Ratios)
  - Length-based features (absolute difference, average length, longest common substring)
- **Machine Learning Model** – Trained **Random Forest Classifier**
- **Web Interface** – User-friendly **Flask app** with HTML/CSS frontend
- **Real-Time Prediction** – Instant similarity checking through web form

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone <https://github.com/CARLOX62/Duplicate-Questions-Pair>
cd <project-directory>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK Stopwords
```python
import nltk
nltk.download('stopwords')
```

---

## 🧪 Usage

### ▶️ Run the Web Application
```bash
python app.py
```

Then open your browser and go to:  
👉 `https://duplicate-questions-pair-2.onrender.com`

Enter two questions and click **"Check Similarity"** to see if they are *Duplicate* or *Not Duplicate*.

---

## 🧰 Using the Model Directly
```python
import pickle
from app import query_point_creator

# Load model and vectorizer
rf = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))

# Example questions
q1 = "What is the capital of France?"
q2 = "What is Paris?"

query = query_point_creator(q1, q2)
pred = rf.predict(query)[0]
print("Duplicate" if pred == 1 else "Not Duplicate")
```

---

## 🧠 Model Details
- **Algorithm**: Random Forest Classifier  
- **Features**: 22 engineered features + Bag of Words vectors  
- **Dataset**: Question pairs dataset (`Questions.csv`)  
- **Preprocessing Includes**:
  - Lowercasing & stripping  
  - Number abbreviation expansion (k, m, b)
  - Currency symbol replacement  
  - Contraction expansion  
  - HTML tag & punctuation removal  

---

## 📂 Project Structure
```
├── app.py                              # Main Flask app
├── requirements.txt                    # Dependencies
├── model.pkl                           # Trained Random Forest model
├── cv.pkl                              # CountVectorizer model
├── Questions.csv                       # Dataset
├── templates/
│   └── index.html                      # Frontend HTML
├── Advanced features with BOW.ipynb    # Feature engineering notebook
├── BAG of Words.ipynb                  # Basic BOW implementation
├── BOW With some features.ipynb        # Feature combination notebook
├── EDA.ipynb                           # Exploratory data analysis
└── README.md                           # This file
```

---

## 📦 Dependencies
```
Flask==3.1.0
gunicorn==23.0.0
numpy==2.2.4
pandas==2.2.3
beautifulsoup4==4.14.2
nltk==3.9.2
Distance==0.1.3
fuzzywuzzy==0.18.0
scikit-learn==1.6.1
```

---

## 🤝 Contributing
1. Fork the repository  
2. Create a new branch  
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes  
   ```bash
   git commit -m "Add some AmazingFeature"
   ```
4. Push to the branch  
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a **Pull Request**

---

## 🪪 License
This project is licensed under the **MIT License** — see the `LICENSE` file for details.

---

## 💡 Author
Developed by **Aniket Kumar** — AI/ML Developer & Student Passionate About NLP and Web Integration 🚀
