# Iris Flower Classification

This project was built for the CodeAlpha Data Science Internship. The goal is to classify Iris flowers into three species (Setosa, Versicolor, and Virginica) based on sepal and petal measurements.

## Dataset
The dataset contains 150 samples with four numerical features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)
- Species (Target: Setosa, Versicolor, Virginica)

## Methodology
1. **Exploratory Data Analysis**: Checked feature distributions, pairwise relationships, and correlations.
2. **Preprocessing**: Split the data into 80% training and 20% testing sets using stratified sampling, then scaled the features with `StandardScaler`.
3. **Model Training & Comparison**: Evaluated multiple classifiers including Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, and Support Vector Machine (SVM).
4. **Validation**: Used 5-fold cross-validation on the training set and evaluated on the unseen test set.

## Results
- **Support Vector Machine (SVM)** gave the best performance with **100% test accuracy** and a 5-fold cross-validation score of **97.5%**.
- Key plots generated during analysis are stored in the `plots/` folder:
  - `pairplot.png`: Pairwise feature distributions by species.
  - `correlation_heatmap.png`: Correlation matrix of measurements.
  - `confusion_matrix.png`: Prediction performance matrix.

## Project Structure
```
├── plots/
│   ├── pairplot.png
│   ├── correlation_heatmap.png
│   └── confusion_matrix.png
├── iris_classification.py
├── Iris_Flower_Classification.ipynb
├── iris_dataset.csv
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python iris_classification.py
```
You can also run through `Iris_Flower_Classification.ipynb` using Jupyter Notebook or Google Colab.
