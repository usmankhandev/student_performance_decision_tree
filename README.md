# Student Digital Life & Academic Performance Prediction

## Project Overview

This project explores how students' **digital habits** affect their academic performance using **Decision Tree** algorithm (Supervised Learning).

We solved two problems:
- **Regression**: Predicting exact `final_exam_score`
- **Classification**: Classifying students as `Poor`, `Average`, or `Good` performers

---

## Dataset Information

- **Filename**: `student_digital_life.csv`
- **Rows**: 15,000+
- **Target Variable**: `final_exam_score` (0-100)
- **Key Features**: Study hours, smartphone usage, social media, gaming, sleep, attendance, mental health, etc.

---

## Technologies

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib
- graphviz

---

## How to Run the Project

### Step 1: Install Dependencies

```bash
pip install pandas numpy scikit-learn matplot lib graphviz ```


### Step 1: Install Dependencies

# Ubuntu / Linux
sudo apt update
sudo apt install graphviz

# macOS
brew install graphviz

python student_performance.py

**Output Files**

After running the script, the following files will be generated:
1. Model Performance

Regression: RMSE and R² Score
Classification: Accuracy + Full Report
Top 10 Feature Importance

decision_tree_regressor.png (Simple Regressor Tree image)

decision_tree_classifier.png (Simple Classifier Tree image)

decision_tree_regressor.svg (High-quality zoomable Regressor Tree vector)

decision_tree_classifier.svg (High-quality zoomable Classifier Tree vector)





