# 🎓 Students' Academic Performance Predictor

An interactive, end-to-end Machine Learning web application that predicts student academic performance levels based on behavioral patterns, classroom engagement metrics, and demographic features. 

The core predictive engine leverages a **K-Nearest Neighbors (KNN)** classifier trained on educational data, seamlessly deployed via a user-friendly web interface.

🚀 **Live Web App:** [View Deployed App on Streamlit](https://studend--academic-performance-knn-9phv7gkuj7ih5a5lcxcutj.streamlit.app/)

---

## 📊 Dataset Overview
The project utilizes the **xAPI-Edu-Data** (Students' Academic Performance Dataset), collected from a Learning Management System (LMS). It categorizes students into three academic performance brackets based on final marks:
* 🟢 **High-Level (H):** 90–100
* 🟡 **Middle-Level (M):** 70–89
* 🔴 **Low-Level (L):** 0–69

### Key Features Tracked:
* **Academic Behavior:** Raised hands, Visited resources, Viewing announcements, Discussion group participation.
* **Demographics & Background:** Gender, Nationality, Educational Stage, Grade Level.
* **Attendance:** Student absence days (Above/Under 7 days).

---

## 🛠️ Tech Stack & Libraries
* **Frontend UI:** Streamlit (Python-based Web Framework)
* **Machine Learning:** Scikit-learn (KNN Classifier)
* **Data Processing:** Pandas, NumPy
* **Model Serialization:** Joblib

---

## ⚙️ Local Installation & Setup

If you want to run this project locally on your machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/shriyan-h/studend--academic-performance-KNN.git](https://github.com/shriyan-h/studend--academic-performance-KNN.git)
   cd studend--academic-performance-KNN