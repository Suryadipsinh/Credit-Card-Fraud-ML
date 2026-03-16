# Credit Card Fraud ML Workflow

**Author:** Suryadipsinh Vaghela

## 📌 Project Overview
Financial security is more critical today than ever before. The primary objective of this project is to accurately predict whether a credit card transaction is fraudulent. By building a reliable machine learning detection model, credit card companies can proactively protect their customers and ensure they are not charged for unauthorized purchases. 

The primary challenge in this project is that the dataset is extremely unbalanced: only 492 out of 284,807 transactions are fraudulent. This means the positive class (frauds) accounts for a mere **0.172%** of all transactions, requiring specialized techniques (like SMOTE) to train the model effectively.

## 📊 Dataset Information
The dataset used in this project can be found on Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud).

Due to confidentiality issues, the original features of the dataset are not provided. Instead, most of the features are numerical input variables that are the result of a Principal Component Analysis (PCA) transformation. 

* **Features V1, V2, ... V28:** The principal components obtained using PCA. These are the only transformed features.
* **Time:** The seconds elapsed between each transaction and the first transaction in the dataset.
* **Amount:** The total transaction amount.
* **Class:** The response variable. It takes a value of `1` in the case of fraud, and `0` otherwise.

## ⚙️ Methodology: How & Why

To tackle this problem, an end-to-end Machine Learning workflow was implemented:

### 1. Handling Imbalanced Data (The "Why")
Standard machine learning algorithms struggle with highly imbalanced datasets. If a model simply predicts "Not Fraud" for every transaction, it would be 99.8% accurate, but it would catch **zero** actual fraud cases. 
* **The "How":** To fix this, I used the **SMOTE (Synthetic Minority Over-sampling Technique)** algorithm to upsample the minority class (fraudulent cases). This generates synthetic examples of fraud so the model has enough data to learn the patterns of fraudulent behavior.

### 2. Model Training & Selection
I trained and evaluated four different machine learning models to find the best fit for this specific problem:
1. **Logistic Regression** (Used as the Baseline Model)
2. **Decision Tree**
3. **Random Forest**
4. **XGBoost**

*Hyperparameter tuning was conducted using Grid Search Cross-Validation to optimize the Random Forest and XGBoost models.*

### 3. Evaluation Metrics
In fraud detection, Accuracy is a poor metric. Instead, I evaluated the models based on:
* **Precision:** Out of all the transactions the model flagged as fraud, how many were *actually* fraud? (Minimizes false positives).
* **Recall:** Out of all the *actual* fraud transactions, how many did the model successfully find? (Minimizes false negatives).
* **Confusion Matrix:** Visualized using a custom helper script (`plot_cm.py`) to see exact True/False Positive/Negative distributions.

## 🚀 Results & Conclusion

After training and evaluating the models, **Ensemble methods (Random Forest and XGBoost) significantly outperformed the simpler models (Logistic Regression and Decision Tree).** They separated the data much better and exhibited a much smaller precision-recall trade-off.

### Key Performance Highlights:
* **XGBoost** improved **recall by 7.9%** and **precision by 23.3%** compared to the baseline Logistic Regression model.
* **Precision Ranking:** Random Forest > XGBoost > Logistic Regression > Decision Tree.
* **Recall Ranking:** XGBoost > Random Forest > Decision Tree = Logistic Regression.

### Trade-offs & Final Thoughts:
While the actual training time for all models was relatively short (30s to 1 min per fit), tuning the heavy array of hyperparameters for Random Forest and XGBoost took significantly more computational time compared to the simpler models. However, the massive 20%+ improvement in precision makes the extra computational cost of XGBoost/Random Forest highly worthwhile for real-world credit card fraud detection.

## 📁 Repository Structure

* `FraudDetection.ipynb`: The main Jupyter Notebook containing the full EDA, SMOTE application, model training, and evaluation code.
* `plot_cm.py`: A Python helper script containing a customized function to nicely print and plot the confusion matrix.
* `README.md`: Project documentation.

## 🛠️ How to Run
1. Clone the repository.
2. Download the dataset from Kaggle and place `creditcard.csv` in the root directory.
3. Install required libraries: `pip install pandas numpy scikit-learn matplotlib xgboost imbalanced-learn`
4. Run all cells in `FraudDetection.ipynb`.