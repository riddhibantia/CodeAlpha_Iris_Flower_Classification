import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Classification models to compare
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def main():
    print("=" * 60)
    print("CodeAlpha Data Science Internship - Task 1: Iris Flower Classification")
    print("=" * 60)

    # 1. Load Dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    # Save dataset copy for record
    data_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(data_dir, "iris_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Dataset loaded and saved to: {csv_path}")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 records:")
    print(df.head())

    # 2. Basic Statistical Summary
    print("\nSummary Statistics:")
    print(df.describe().T)

    print("\nClass distribution:")
    print(df['species'].value_counts())

    # 3. Visualizations (Saved for GitHub README / LinkedIn showcase)
    plots_dir = os.path.join(data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Pairplot
    sns.set_theme(style="whitegrid")
    pairplot = sns.pairplot(df, hue="species", markers=["o", "s", "D"], palette="viridis")
    pairplot.fig.suptitle("Iris Feature Pairplot by Species", y=1.02)
    pairplot.savefig(os.path.join(plots_dir, "pairplot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(8, 6))
    numeric_df = df.drop(columns=['species'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Iris Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "correlation_heatmap.png"), dpi=300)
    plt.close()
    print(f"\n[+] Plots saved to: {plots_dir}")

    # 4. Data Preparation (Train/Test Split & Scaling)
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Model Evaluation and Comparison
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Support Vector Machine (SVM)": SVC(kernel='linear', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    print("\n" + "=" * 60)
    print("Model Training & Evaluation (Accuracy & 5-Fold Cross-Validation)")
    print("=" * 60)

    best_model_name = None
    best_model = None
    best_acc = 0.0
    results = []

    for name, model in models.items():
        # Train on scaled features
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        results.append({
            "Model": name,
            "Test Accuracy": f"{acc * 100:.2f}%",
            "CV Mean": f"{cv_scores.mean() * 100:.2f}%",
            "CV Std": f"{cv_scores.std() * 100:.2f}%"
        })
        
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model = model

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # 6. Detailed Performance for the Best Model
    print("\n" + "=" * 60)
    print(f"Best Performing Model: {best_model_name} (Test Accuracy: {best_acc * 100:.2f}%)")
    print("=" * 60)

    y_pred_best = best_model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best, target_names=target_names))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.ylabel('Actual Species')
    plt.xlabel('Predicted Species')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # 7. Sample Prediction Demo
    print("=" * 60)
    print("Sample Inference Demonstration")
    print("=" * 60)
    sample_inputs = np.array([
        [5.1, 3.5, 1.4, 0.2],  # Expected: setosa
        [6.0, 2.7, 4.0, 1.3],  # Expected: versicolor
        [6.5, 3.0, 5.5, 2.0]   # Expected: virginica
    ])
    sample_inputs_scaled = scaler.transform(sample_inputs)
    preds = best_model.predict(sample_inputs_scaled)

    for i, (features, pred) in enumerate(zip(sample_inputs, preds)):
        print(f"Sample {i+1} Features {features.tolist()} -> Predicted Species: {target_names[pred]}")

    print("\n[OK] Task 1 execution completed successfully!")

if __name__ == "__main__":
    main()
