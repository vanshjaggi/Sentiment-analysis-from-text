# 🧠 Sentiment Analysis from Text

A full-stack AI/ML web application that predicts the sentiment of a given text (Positive or Negative) using a trained Machine Learning model. The app is built with:

- 🔮 **scikit-learn** for training the sentiment analysis model
- 🧠 **Flask** as the backend API
- 🌐 **Streamlit** as the frontend UI
- 🧪 **Python virtual environment** to isolate dependencies

---

## 📁 Project Structure

```
SENTIMENT-ANALYSIS-APP/
│
├── frontend/
│   └── app.py                  # Streamlit UI
│
├── backend/
│   └── app.py                  # Flask API for predictions
│
├── model/
│   ├── train_model.py          # Script to train and save the model
│   └── sentiment_model.pkl     # Pre-trained sentiment model
│
├── data/
│   └── imdb.csv                # Training data (IMDB movie reviews)
│
├── requirements.txt            # Python package dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Technologies Used

| Purpose     | Tools/Libraries              |
|-------------|------------------------------|
| ML Model    | `scikit-learn`, `joblib`     |
| Backend API | `Flask`                      |
| Frontend UI | `Streamlit`                  |
| Language    | `Python 3`                   |
| Dev Tools   | `venv`, `Git`  |

---

## 🤖 Model Details

- **AI Type**: Supervised Learning (Classification)
- **Model**: Multinomial Naive Bayes
- **Text Vectorizer**: TF-IDF
- **Task**: Classify sentiment as **Positive** or **Negative**
- **Training Data**: IMDB Movie Review Dataset

### 🛠️ Training Pipeline

```python
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('nb', MultinomialNB())
])
```

The model is trained using `train_model.py` and saved as `sentiment_model.pkl`.

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/SENTIMENT-ANALYSIS-APP.git
cd SENTIMENT-ANALYSIS-APP
```

### 2️⃣ Create and Activate Virtual Environment

```bash
python -m venv venv
```

- **Windows**: `venv\Scripts\activate`
- **Linux/macOS**: `source venv/bin/activate`

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train the Model (If not already)

```bash
python model/train_model.py
```

### 5️⃣ Run the Application

```bash
python backend/app.py

python frontend/app.py

(in different command line windows)
```
## 🔁 API Endpoint (Flask)

**POST** `/predict`

**Request Body:**

```json
{
  "text": "This movie was absolutely amazing!"
}
```

**Response:**

```json
{
  "sentiment": "positive"
}
```

---

## 💡 Features

- Real-time text sentiment prediction
- Model training script included
- Clean Streamlit user interface
- Flask backend API for modularity

---

## ✅ To-Do / Future Improvements

- Add more sentiment categories (neutral, mixed, etc.)
- Use deep learning (LSTM, BERT)
- Save user query history or feedback
- Host app using Streamlit Cloud or Render
- Add support for multi-language sentiment detection

---

## 👨‍💻 Author

Made by **Vansh Jaggi**
