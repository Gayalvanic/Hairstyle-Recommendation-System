# 💇 Hairstyle Recommendation System

A machine learning-based system that recommends hairstyles based on a user's facial features. It detects facial shape using supervised ML and maps it to the most suitable hairstyle.

---

## 🧠 How It Works

1. User inputs facial measurement data
2. The system extracts 9 key facial features from the dataset
3. A **Random Forest Classifier** predicts the user's face shape
4. The system returns a hairstyle recommendation based on the predicted shape

### Face Shapes Detected
- 🫀 Heart
- 🟥 Square
- 🔵 Round
- 🥚 Oval

---

## 🔬 ML Pipeline

| Step | Details |
|------|---------|
| Dataset | Custom CSV dataset (`dataset27.csv`) — 81 samples, 9 facial features |
| Preprocessing | `StandardScaler` for feature normalization |
| Model | `RandomForestClassifier` (n_estimators=50) |
| Validation | `K-Fold Cross Validation` (k=20) |
| Selection | Best fold by maximum accuracy score used for final prediction |

> Earlier experiments also used **Logistic Regression** before settling on Random Forest for improved accuracy.

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:**
  - `pandas` — data loading and manipulation
  - `numpy` — numerical operations
  - `scikit-learn` — ML model, scaling, cross-validation
- **Dataset:** Custom CSV with facial landmark measurements

---

## 📁 Project Structure

```
Hairstyle-Recommendation-System/
│
├── Code.py              # Logistic Regression experiment
├── Code1.py             # Final model using Random Forest
├── Prediction.py        # Prediction logic
├── User_Input.py        # Handles user input
├── User_Point.py        # Facial point extraction
├── app.py               # Main application entry point
├── Main.ipynb           # Jupyter Notebook exploration
├── Untitled.ipynb       # Additional experiments
├── dataset27.csv        # Training dataset
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn
```

### Run the App
```bash
python app.py
```

---

## 📊 Model Performance

- Cross-validation across **20 folds**
- Final prediction selected from the fold with **maximum accuracy**
- Handles multiclass classification (4 face shape categories)

---

## 🔮 Future Improvements

- Integrate real-time **webcam facial detection** using OpenCV
- Expand dataset for improved generalization
- Add a web-based UI using Flask or Streamlit
- Include hairstyle images in recommendations

---

## 👩‍💻 Author

**Gayalvanic**  
[GitHub Profile](https://github.com/Gayalvanic)
