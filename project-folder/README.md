# Adult Income Classification Project

## a. Problem Statement

The objective of this project is to develop machine learning classification models to predict whether an individual's annual income exceeds $50K per year based on demographic and employment-related features.

This is a binary classification problem:
- Class 0 → Income ≤ 50K
- Class 1 → Income > 50K

The goal is to compare multiple ML models using standard evaluation metrics and determine the best-performing model.

---

## b. Dataset Description  [1 Mark]

The dataset used is the Adult Income dataset, which contains demographic and employment information of individuals.

Key Features include:
- Age
- Workclass
- Education
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Capital Gain
- Capital Loss
- Hours per Week
- Native Country

Target Variable:
- Income (≤50K / >50K)

The dataset consists of both numerical and categorical features. Preprocessing steps such as encoding categorical variables and scaling numerical features were performed before training the models.

---

## c. Models Used and Evaluation Metrics  [6 Marks]

The following six machine learning models were implemented:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Naive Bayes
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

Each model was evaluated using:
- Accuracy
- AUC (Area Under ROC Curve)
- Precision
- Recall
- F1 Score
- MCC (Matthews Correlation Coefficient)

---

## Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|----------|-----------|----------|----------|----------|
| Logistic Regression | 0.852975 | 0.907738 | 0.756994 | 0.618954 | 0.681050 | 0.591786 |
| Decision Tree | 0.812531 | 0.757259 | 0.626667 | 0.645098 | 0.635749 | 0.509668 |
| kNN | 0.813194 | 0.831738 | 0.653465 | 0.560784 | 0.603588 | 0.484737 |
| Naive Bayes | 0.450191 | 0.694940 | 0.310899 | 0.960131 | 0.469704 | 0.250291 |
| Random Forest (Ensemble) | 0.850820 | 0.902255 | 0.736486 | 0.641176 | 0.685535 | 0.590791 |
| XGBoost (Ensemble) | 0.870214 | 0.927999 | 0.781038 | 0.678431 | 0.726128 | 0.644366 |

---

## Observations on Model Performance  [3 Marks]

| ML Model Name | Observation about Model Performance |
|---------------|---------------------------------------|
| Logistic Regression | Strong baseline model with high AUC (0.9077) and balanced precision-recall performance. |
| Decision Tree | Moderate accuracy but lower AUC. May suffer from overfitting. |
| kNN | Similar accuracy to Decision Tree but lower recall. Sensitive to feature scaling and choice of k. |
| Naive Bayes | Very high recall (0.9601) but extremely low precision and accuracy, indicating many false positives. |
| Random Forest (Ensemble) | Improved generalization over Decision Tree with strong AUC and balanced metrics. |
| XGBoost (Ensemble) | Best overall performance with highest Accuracy, AUC, F1 Score, and MCC. Recommended model. |

---

## Final Conclusion

Among all six models, XGBoost achieved the best overall performance across all evaluation metrics. 
Ensemble methods (Random Forest and XGBoost) outperformed individual classifiers on this dataset.
XGBoost is recommended for deployment for the Adult Income classification task.
