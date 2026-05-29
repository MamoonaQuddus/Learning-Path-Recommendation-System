# 🎓 Learning Path Recommendation System

A Machine Learning-based recommendation system that suggests personalized learning courses and generates structured learning paths for interns using **Collaborative Filtering** and **Matrix Factorization (Truncated SVD)**.

---

# 📌 Project Overview

This project analyzes intern-course interaction data and predicts which courses an intern is most likely to prefer based on historical learning patterns.

The system also creates intelligent learning paths by organizing recommended courses according to:
- Difficulty progression
- Course duration

The project includes:
- Recommendation Engine
- Learning Path Generator
- Streamlit Web Application
- Matrix Factorization Model

---

# 🚀 Features

- Personalized course recommendations
- Learning path generation
- Collaborative filtering
- Matrix factorization using Truncated SVD
- Difficulty-based filtering
- Category-based filtering
- Interactive Streamlit dashboard
- Scalable object-oriented architecture

---

# 🧠 Machine Learning Technique

This project uses:

## Collaborative Filtering

with:

## Matrix Factorization (Truncated SVD)

The interaction matrix is decomposed into latent factors:

\[
R \approx P Q^T
\]

Where:
- \(R\) = Intern-Course Interaction Matrix
- \(P\) = Intern Latent Features
- \(Q\) = Course Latent Features

The model learns hidden learning patterns such as:
- AI/ML interests
- Web development preferences
- Beginner vs advanced learner behavior

---

# 📂 Dataset Structure

## 1. Course Metadata Dataset

### `course_metadata.csv`

### Example

| course_id | title | category | difficulty | duration_hours |
|---|---|---|---|---|
| C001 | Python Basics | AI/ML | Beginner | 20 |

---

## 2. Intern Ratings Dataset

### `intern_ratings.csv`

### Example

| intern_id | course_id | rating |
|---|---|---|
| 1 | C001 | 5 |

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Computation |
| Scikit-learn | Machine Learning |
| Streamlit | Web Application |
| TruncatedSVD | Matrix Factorization |

---

# 📦 Installation

Install required libraries:

```bash
pip install pandas numpy scikit-learn streamlit
```

---

# ▶️ Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

---

# 📖 Usage

## Step 1 — Upload Datasets

Upload:
- `course_metadata.csv`
- `intern_ratings.csv`

through the Streamlit sidebar.

---

## Step 2 — Select Recommendation Settings

Choose:
- Intern ID
- Number of recommendations
- Difficulty filter
- Category filter
- Learning path strategy

---

## Step 3 — Generate Recommendations

The system predicts personalized courses for the selected intern.

---

## Step 4 — Generate Learning Path

The system creates an ordered learning sequence.

---

# 🛤️ Learning Path Strategies

## 1. Difficulty Ascending

Orders courses as:

```text
Beginner → Intermediate → Advanced
```

---

## 2. Shortest First

Orders courses by:

```text
Lowest duration → Highest duration
```

---

# 📊 Example Output

| course_id | title | category | difficulty | predicted_score |
|---|---|---|---|---|
| C003 | Deep Learning | AI/ML | Advanced | 0.94 |
| C021 | AWS Cloud | Cloud | Intermediate | 0.91 |

---

---

# 🔮 Future Improvements

- Hybrid Recommendation Systems
- Deep Learning Recommendation Models
- Real-time Recommendation Updates
- Course Completion Tracking
- User Authentication
- Web Deployment
- Content-Based Filtering
- Reinforcement Learning

---

# 🌍 Applications

This project can be used in:
- E-learning platforms
- Internship programs
- Corporate training systems
- Personalized education platforms
- Online academies

---

# ✅ Conclusion

This project demonstrates how Machine Learning and Collaborative Filtering can be used to build personalized educational recommendation systems.

The recommendation engine intelligently predicts intern interests and generates structured learning paths using Matrix Factorization and latent feature learning.

The project also demonstrates:
- Recommendation Systems
- Matrix Factorization
- Object-Oriented Programming
- Data Processing
- Scalable ML Architecture
- Interactive ML Applications

---
