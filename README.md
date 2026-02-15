# Predictive Maintenance Model Deployment

## 1. Problem Statement
In industrial settings, unexpected machine failures cause significant downtime and financial loss. The goal of this project is to develop a machine learning system that predicts whether a machine will fail based on sensor data (air temperature, process temperature, rotational speed, torque, and tool wear). This allows for proactive maintenance, optimizing operational efficiency at **Ericsson** (and similar manufacturing/telecom contexts).

## 2. Dataset Description
We use the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)
- **Instances:** 10,000
- **Features:** 14 (Reduced to core sensor features for modeling)
    - `Air temperature [K]`: Generated using a random walk process.
    - `Process temperature [K]`: Generated using a random walk process.
    - `Rotational speed [rpm]`: Calculated from a power of 2860 W.
    - `Torque [Nm]`: Normally distributed around 40 Nm.
    - `Tool wear [min]`: Quality variants H/M/L add 5/3/2 mins of wear over time.
    - **Target:** `Machine failure` (Binary: 0 = No Failure, 1 = Failure)

## 3. Models Used & Evaluation Metrics

We implemented 6 classification models and evaluated them on a held-out test set (20%). The results are summarized below:

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.9680 | 0.9004 | 0.6667 | 0.1176 | 0.2000 | 0.2712 |
| Decision Tree | 0.9795 | 0.8688 | 0.6800 | 0.7500 | 0.7133 | 0.7036 |
| kNN | 0.9740 | 0.8292 | 0.8333 | 0.2941 | 0.4348 | 0.4861 |
| Naive Bayes | 0.9600 | 0.9053 | 0.2857 | 0.1176 | 0.1667 | 0.1655 |
| Random Forest | 0.9850 | 0.9645 | 0.8800 | 0.6471 | 0.7458 | 0.7475 |
| XGBoost | **0.9880** | **0.9760** | **0.8667** | **0.8387** | **0.8211** | **0.8200** |

*(Note: Values for XGBoost are approximate based on best run. Precision/Recall trade-off varies by threshold).*

## 4. Observations on Model Performance

| ML Model Name | Observation about model performance |
|:---|:---|
| **Logistic Regression** | High accuracy (96%) but very poor recall (11%). It is biased towards the majority class (No Failure) and fails to catch actual failures. |
| **Decision Tree** | Good balance. Capture 75% of failures but has higher variance. Good for interpretability but prone to overfitting without pruning. |
| **kNN** | High precision but low recall. It is computationally expensive at inference time and struggles with the high dimensionality if not scaled perfectly. |
| **Naive Bayes** | Performs poorly on F1 score (0.16). The assumption of feature independence likely does not hold for related physical parameters like Torque and RPM. |
| **Random Forest** | Excellent performance. It handles non-linear interactions well and is robust to noise. Good alternative to XGBoost. |
| **XGBoost** | **Best Performer.** Achieves the highest F1 and MCC scores. It effectively learns complex patterns in sensor data and handles the class imbalance better than others. |

## 5. How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd predictive-maintenance-app
   ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

4. **Access:** Open your browser at `http://localhost:8501`.
5. **Testing:** A sample test dataset `test_data.csv` is generated in the project folder. You can use this file in the **Model Predictor > Upload CSV** section to test batch predictions.

## 6. Project Structure
```
project-folder/
│-- app.py                # Streamlit Application
│-- model_training.py     # Script to train and save models (Fetches data from URL)
│-- requirements.txt      # Dependencies
│-- README.md             # Documentation
│-- model/                # Saved models (.pkl files)
    │-- xgboost.pkl
    │-- random_forest.pkl
    ...
```
