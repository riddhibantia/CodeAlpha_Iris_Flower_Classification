# CodeAlpha Data Science Internship - Task 1: Iris Flower Classification

## 📌 Project Overview
This project is completed as part of the **CodeAlpha Data Science Internship Program**. The goal is to build a machine learning model capable of classifying Iris flower species (*Setosa, Versicolor, and Virginica*) based on sepal and petal measurements (length and width).

---

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Manipulation & Analysis:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`

---

## 🔬 Dataset Description
The Iris dataset consists of **150 samples** across **3 classes** (50 samples per class):
1. **Iris-setosa**
2. **Iris-versicolor**
3. **Iris-virginica**

### Features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

---

## 📊 Exploratory Data Analysis & Visualizations
The script generates and saves the following visualizations in `plots/`:
- **`pairplot.png`**: Multi-dimensional feature scatter plots highlighting class separability.
- **`correlation_heatmap.png`**: Correlation analysis showing strong relationships between petal dimensions and species differentiation.
- **`confusion_matrix.png`**: Detailed true vs. predicted species performance matrix.

---

## 🤖 Models Trained & Evaluated
Multiple supervised classification algorithms were trained, benchmarked, and validated using 5-fold cross-validation:
1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Decision Tree Classifier**
4. **Support Vector Machine (SVM)**
5. **Random Forest Classifier**

---

## 🚀 How to Run
```bash
# 1. Navigate to the task directory
cd Task_1_Iris_Flower_Classification

# 2. Run the classification script
python iris_classification.py
```
