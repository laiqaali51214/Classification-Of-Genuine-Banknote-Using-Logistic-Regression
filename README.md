# Classification-Of-Genuine-Banknote-Using-Logistic-Regression

## 📌 Overview
This project focuses on classifying genuine vs. counterfeit banknotes using machine learning. The dataset contains features extracted from wavelet transforms of banknote images, and a logistic regression model is trained to detect authenticity.

🔗 **Dataset Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/static/public/267/data.csv)

---

## 🛠️ Technical Stack
- **Python Libraries**:  
  `numpy`, `pandas`, `seaborn`, `matplotlib`, `scikit-learn`, `statsmodels`
- **Machine Learning Model**: Logistic Regression
- **Validation**: 5-Fold Cross Validation
- **Key Metrics**: Accuracy, F1-Score, Confusion Matrix

---

## 📊 Dataset Description
**Features**:
1. **Variance** (of wavelet-transformed image)
2. **Skewness** (of wavelet-transformed image)
3. **Curtosis** (of wavelet-transformed image)
4. **Entropy** (of image)
5. **Class** (0 = genuine, 1 = counterfeit)

**Samples**: 1,372 initial entries → 1,348 after duplicate removal

---

## 🧩 Project Workflow

### 1. Data Preparation & Cleaning
- Removed 24 duplicate entries
- Checked for missing values (none found)
- Analyzed feature distributions and correlations
- Visualized data with pairplots and correlation matrices

### 2. Exploratory Data Analysis

### 3. Model Training
- **Logistic Regression** with L2 regularization
- 80-20 train-test split
- 5-fold cross-validation

### 4. Evaluation Metrics
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 99.26% |
| F1-Score  | 99.23% |
| Precision | 99.35% |
| Recall    | 99.12% |

**Confusion Matrix**:
```
[[155   1]
 [  1 116]]
```

---

## 🔍 Key Insights
- **Skewness** and **Curtosis** showed highest correlation with authenticity
- Dataset was well-balanced (55% genuine vs 45% counterfeit)
- Model achieved near-perfect classification with 99%+ accuracy
- Minimal misclassifications (2 errors in test set)

---

## 💡 Future Improvements
- Experiment with neural networks/ensemble methods
- Implement feature engineering
- Test anomaly detection approaches
- Deploy as API for real-time verification
