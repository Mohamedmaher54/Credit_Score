# Credit Score Prediction: Project Documentation

## Project Goal
The primary objective of this project was to develop a machine learning model to predict a customer's credit score ('Good', 'Standard', or 'Poor') based on their historical financial and personal data over an eight-month period.

---

## Data Structure and Modeling Approach

- **Dataset:** The training dataset consists of 100,000 rows, with each customer represented by 8 rows—one for each month. The target variable is the credit score for each month.
- **Objective:** Predict the monthly credit score for each customer based on their historical financial and personal data.
- **Time-Series Nature:** Since each customer has multiple records over time, the data is inherently time-series and requires careful handling to avoid data leakage.

### Train/Test Split Strategy

- **Initial Approach:**  
  - For each customer, the first 6 months were used for training, and the last 2 months were used for validation/testing.
  - This simulates a real-world scenario where we predict future credit scores based on past data, and prevents data leakage.
  - This approach achieved an accuracy of approximately 72%.

- **Final Model for Deployment:**  
  - After validating the approach, the model was retrained on the complete dataset (all 8 months for each customer) to maximize the use of available data.
  - The final model was then used to predict credit scores on the test set, and this version was deployed.

---

## 1. Initial Data Exploration (EDA)
The project began with a thorough exploration of the provided dataset (`train.csv`), which contained 100,000 records across 28 columns.

**Key Findings from EDA:**
- **Data Structure:** The dataset is a time-series, with each customer having a record for each of the eight months. This structure was critical for the modeling approach.
- **Target Variable Distribution:** The `Credit_Score` target variable was found to be imbalanced, with 'Standard' being the most frequent category. This imbalance can affect model training and evaluation.
  - (Placeholder for the Credit_Score count plot from the notebook)
- **Data Quality Issues:** A significant portion of the project involved identifying and addressing numerous data quality problems:
  - Inconsistent Data Types: Many numerical columns like `Annual_Income` and `Age` were incorrectly stored as 'object' types due to inconsistent formatting.
  - Formatting Errors: Widespread use of placeholders (`_______`) and random underscores (`_`) in numerical and categorical columns.
  - Missing Data: Several columns had a substantial number of null values.
  - Outliers: Extreme and illogical values were present, such as `Num_Bank_Accounts` having a value of 1294.
- **Feature Distributions:**
  - Categorical features like `Occupation`, `Credit_Mix`, and `Payment_Behaviour` showed varying distributions, suggesting they could be valuable predictors.
  - Numerical features like `Monthly_Inhand_Salary` and `Interest_Rate` had skewed distributions, indicating the presence of outliers.

---

## 2. Baseline Modeling (Minimal Cleaning)

Before building a robust pipeline, a baseline model was created using only the minimal data cleaning required for the models to run. Three models were trained and evaluated:

- **RandomForest**
- **XGBoost**
- **LightGBM**

**Results:**

--- Training RandomForest ---

Accuracy for RandomForest: **0.7017**

```
Classification Report for RandomForest:
              precision    recall  f1-score   support

        Good       0.65      0.60      0.62      4824
        Poor       0.71      0.72      0.71      7216
    Standard       0.72      0.73      0.72     12960

    accuracy                           0.70     25000
   macro avg       0.69      0.68      0.69     25000
weighted avg       0.70      0.70      0.70     25000
```

--- Training XGBoost ---

Accuracy for XGBoost: **0.6828**

```
Classification Report for XGBoost:
              precision    recall  f1-score   support

        Good       0.62      0.60      0.61      4824
        Poor       0.69      0.67      0.68      7216
    Standard       0.70      0.72      0.71     12960

    accuracy                           0.68     25000
   macro avg       0.67      0.67      0.67     25000
weighted avg       0.68      0.68      0.68     25000
```

--- Training LightGBM ---

Accuracy for LightGBM: **0.6532**

```
Classification Report for LightGBM:
              precision    recall  f1-score   support

        Good       0.56      0.58      0.57      4824
        Poor       0.67      0.63      0.65      7216
    Standard       0.68      0.69      0.69     12960

    accuracy                           0.65     25000
   macro avg       0.64      0.64      0.64     25000
weighted avg       0.65      0.65      0.65     25000
```

---

## 3. Statistical Tests for Feature Selection

After the baseline, statistical tests were performed to assess feature relevance:

--- Performing Statistical Tests for Feature Selection ---

### ANOVA F-test results (Numerical vs Target)
A low P-Value (< 0.05) suggests the feature is likely correlated with the target.

| Feature                  | F-Score      | P-Value         |
|--------------------------|--------------|-----------------|
| Monthly_Inhand_Salary    | 1535.70      | 0.000000e+00    |
| Delay_from_due_date      | 9521.54      | 0.000000e+00    |
| Credit_Utilization_Ratio | 93.19        | 3.78e-41        |
| Num_Credit_Inquiries     | 5.48         | 4.17e-03        |
| Num_Bank_Accounts        | 3.49         | 3.06e-02        |
| Num_Credit_Card          | 2.96         | 5.17e-02        |
| Interest_Rate            | 2.39         | 9.13e-02        |
| Total_EMI_per_month      | 0.45         | 6.40e-01        |

### Chi-squared test results (Categorical vs Target)
A low P-Value (< 0.05) suggests the feature is likely correlated with the target.

| Feature                  | Chi2-Score   | P-Value         |
|--------------------------|--------------|-----------------|
| Age                      | 2.21e+05     | 0.000000e+00    |
| Annual_Income            | 1.25e+06     | 0.000000e+00    |
| Type_of_Loan             | 1.31e+04     | 0.000000e+00    |
| Num_of_Loan              | 6.37e+05     | 0.000000e+00    |
| Num_of_Delayed_Payment   | 3.12e+05     | 0.000000e+00    |
| Changed_Credit_Limit     | 2.39e+05     | 0.000000e+00    |
| Outstanding_Debt         | 1.29e+06     | 0.000000e+00    |
| Credit_Mix               | 2.49e+03     | 0.000000e+00    |
| Amount_invested_monthly  | 9.14e+03     | 0.000000e+00    |
| Monthly_Balance          | 4.35e+07     | 0.000000e+00    |
| Credit_History_Age       | 1.36e+04     | 0.000000e+00    |
| Payment_of_Min_Amount    | 2.52e+03     | 0.000000e+00    |
| Payment_Behaviour        | 7.76e+02     | 2.51e-169       |
| Occupation               | 1.98e+01     | 5.14e-05        |
| Month                    | 1.19e+00     | 5.52e-01        |

---

## 4. Data Cleaning: Problems and Solutions

After the baseline, a comprehensive data cleaning pipeline was implemented. Each step addressed a specific data quality issue:

- **Underscore Formatting in Numeric Columns:**
  - *Problem:* Numeric columns (e.g., Age, Annual_Income) contained underscores (e.g., '1_000').
  - *Fix:* Removed underscores and converted to numeric types.

- **Text Placeholders in Categorical Columns:**
  - *Problem:* Placeholders like '_______' and '!@9#%8' appeared in categorical columns.
  - *Fix:* Replaced these with NaN for proper imputation.

- **Age Outliers:**
  - *Problem:* Age values outside a reasonable range (18-100) and inconsistencies per customer.
  - *Fix:* For each customer, replaced with the most frequent valid age or the overall median.

- **Outliers in Customer-Consistent Columns:**
  - *Problem:* Columns that should be constant per customer (e.g., Annual_Income, Num_Bank_Accounts) had inconsistent values.
  - *Fix:* For each customer, replaced with the most frequent value or median.

- **Negative Values in Delayed Payments:**
  - *Problem:* Negative values in Num_of_Delayed_Payment and Delay_from_due_date.
  - *Fix:* Converted all values to absolute (positive) numbers.

- **Missing Occupation:**
  - *Problem:* Missing or placeholder values in Occupation.
  - *Fix:* Imputed with the most frequent occupation for each customer.

- **Categorical Feature Formatting:**
  - *Problem:* Trailing underscores in categorical columns (e.g., Credit_Mix).
  - *Fix:* Removed trailing underscores.

- **Missing Payment_Behaviour:**
  - *Problem:* Missing or placeholder values in Payment_Behaviour.
  - *Fix:* Imputed with the most frequent value using SimpleImputer.

- **Credit History Age:**
  - *Problem:* Credit_History_Age stored as text (e.g., '2 Years 3 Months').
  - *Fix:* Converted to total months and filled missing values by propagating the first valid value for each customer.

---

## 5. Model Results After Data Cleaning

The same three models were retrained after applying the full data cleaning pipeline. Results improved across the board:

--- Training RandomForest ---

Accuracy for RandomForest: **0.7226**

```
Classification Report for RandomForest:
              precision    recall  f1-score   support

        Good       0.70      0.63      0.66      4824
        Poor       0.71      0.77      0.74      7216
    Standard       0.74      0.73      0.73     12960

    accuracy                           0.72     25000
   macro avg       0.72      0.71      0.71     25000
weighted avg       0.72      0.72      0.72     25000
```

--- Training XGBoost ---

Accuracy for XGBoost: **0.6986**

```
Classification Report for XGBoost:
              precision    recall  f1-score   support

        Good       0.64      0.62      0.63      4824
        Poor       0.70      0.71      0.71      7216
    Standard       0.72      0.72      0.72     12960

    accuracy                           0.70     25000
   macro avg       0.69      0.68      0.69     25000
weighted avg       0.70      0.70      0.70     25000
```

--- Training LightGBM ---

Accuracy for LightGBM: **0.6670**

```
Classification Report for LightGBM:
              precision    recall  f1-score   support

        Good       0.57      0.60      0.58      4824
        Poor       0.69      0.65      0.67      7216
    Standard       0.69      0.70      0.70     12960

    accuracy                           0.67     25000
   macro avg       0.65      0.65      0.65     25000
weighted avg       0.67      0.67      0.67     25000
```

---

### Feature Importance (RandomForest, Top 20)

| Feature                    | Importance |
|----------------------------|------------|
| Outstanding_Debt           | 0.103      |
| Interest_Rate              | 0.080      |
| Credit_Mix                 | 0.067      |
| Credit_History_Age_Months  | 0.064      |
| Delay_from_due_date        | 0.063      |
| Changed_Credit_Limit       | 0.054      |
| Num_Credit_Inquiries       | 0.047      |
| Num_Credit_Card            | 0.044      |
| Monthly_Balance            | 0.042      |
| Credit_Utilization_Ratio   | 0.040      |
| Amount_invested_monthly    | 0.040      |
| Num_of_Delayed_Payment     | 0.038      |
| Annual_Income              | 0.036      |
| Monthly_Inhand_Salary      | 0.035      |
| Total_EMI_per_month        | 0.034      |
| Type_of_Loan               | 0.032      |
| Num_Bank_Accounts          | 0.031      |
| Age                        | 0.030      |
| Payment_of_Min_Amount      | 0.025      |
| Occupation                 | 0.025      |

---

## 6. Handling Class Imbalance with SMOTE

The target variable (`Credit_Score`) was highly imbalanced in the original data:

| Credit_Score | Proportion |
|--------------|------------|
| Standard     | 0.53174    |
| Poor         | 0.28998    |
| Good         | 0.17828    |

This imbalance motivated the use of SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes in the training set.

To address the class imbalance in the target variable, SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training data. This technique generates synthetic samples for minority classes to balance the dataset.

**SMOTE Implementation:**
```python
from imblearn.over_sampling import SMOTE
rus = SMOTE(sampling_strategy='auto')
X_data_rus, y_data_rus = rus.fit_resample(X_train, y_train_encoded)
```

- **Training features shape:** (120,642, 22)
- **Testing features shape:** (25,000, 22)

**Results after applying SMOTE:**

--- Training RandomForest ---

Accuracy for RandomForest: **0.7104**

```
Classification Report for RandomForest:
              precision    recall  f1-score   support

        Good       0.62      0.71      0.66      4824
        Poor       0.70      0.80      0.74      7216
    Standard       0.77      0.66      0.71     12960

    accuracy                           0.71     25000
   macro avg       0.69      0.72      0.70     25000
weighted avg       0.72      0.71      0.71     25000
```

--- Training XGBoost ---

Accuracy for XGBoost: **0.4434**

```
Classification Report for XGBoost:
              precision    recall  f1-score   support

        Good       0.39      0.89      0.55      4824
        Poor       0.47      0.89      0.62      7216
    Standard       0.82      0.03      0.05     12960

    accuracy                           0.44     25000
   macro avg       0.56      0.60      0.41     25000
weighted avg       0.64      0.44      0.31     25000
```

--- Training LightGBM ---

Accuracy for LightGBM: **0.4486**

```
Classification Report for LightGBM:
              precision    recall  f1-score   support

        Good       0.39      0.85      0.54      4824
        Poor       0.47      0.86      0.60      7216
    Standard       0.77      0.07      0.13     12960

    accuracy                           0.45     25000
   macro avg       0.54      0.59      0.42     25000
weighted avg       0.61      0.45      0.34     25000
```

---

### Feature Importance (RandomForest, Top 20, with SMOTE)

| Feature                    | Importance |
|----------------------------|------------|
| Interest_Rate              | 0.116      |
| Outstanding_Debt           | 0.115      |
| Credit_Mix                 | 0.083      |
| Delay_from_due_date        | 0.066      |
| Credit_History_Age_Months  | 0.065      |
| Num_Credit_Inquiries       | 0.053      |
| Num_of_Delayed_Payment     | 0.045      |
| Changed_Credit_Limit       | 0.045      |
| Payment_of_Min_Amount      | 0.044      |
| Annual_Income              | 0.035      |
| Num_Credit_Card            | 0.033      |
| Monthly_Balance            | 0.033      |
| Monthly_Inhand_Salary      | 0.033      |
| Num_Bank_Accounts          | 0.032      |
| Total_EMI_per_month        | 0.032      |
| Amount_invested_monthly    | 0.031      |
| Credit_Utilization_Ratio   | 0.031      |
| Age                        | 0.029      |
| Occupation                 | 0.023      |
| Month                      | 0.022      |

---

## 7. CatBoost Model with Tuned Parameters

After experimenting with SMOTE and other models, CatBoost was trained with tuned parameters for improved performance. The following configuration was used:

**CatBoost Parameters:**
```python
model = CatBoostClassifier(
    task_type="GPU",
    iterations=3000,
    learning_rate=0.1,
    depth=6,
    cat_features=cat_features_indices,
    random_seed=42,
    eval_metric='Accuracy',
    verbose=100
)
```

**Results:**

Accuracy for CatBoost: **0.7180**

```
Classification Report for CatBoost:
              precision    recall  f1-score   support

        Good       0.68      0.64      0.66      4824
        Poor       0.71      0.75      0.73      7216
    Standard       0.73      0.73      0.73     12960

    accuracy                           0.72     25000
   macro avg       0.71      0.71      0.71     25000
weighted avg       0.72      0.72      0.72     25000
```

---

## 8. Feature Engineering and Final CatBoost Model

To further improve model performance, several feature engineering techniques were applied:

**Feature Engineering Steps:**
- **Customer Aggregates:**
  - `Avg_Delay_Per_Customer`: Mean delay per customer
  - `Max_Delay_Per_Customer`: Max delay per customer
- **Ratio Features:**
  - `EMI_to_Income`: Total EMI per month divided by Monthly Inhand Salary
  - `Debt_to_Income`: Outstanding Debt divided by Monthly Inhand Salary
  - `Utilization_to_Income`: Credit Utilization Ratio multiplied by Monthly Inhand Salary
- **Lag Features:**
  - `Lag1_EMI`: Previous month's EMI per customer
  - `Lag1_Delay`: Previous month's delay per customer
  - `Lag1_Credit_Util`: Previous month's credit utilization ratio per customer
  - (Missing lag values filled with median)
- **Trend Features:**
  - `EMI_Trend`: Change in EMI from previous month
- **Interaction Features:**
  - `Debt_Utilization`: Outstanding Debt multiplied by Credit Utilization Ratio
  - `Loans_per_Card`: Number of loans divided by number of credit cards
- **Missing Value Handling:**
  - All remaining nulls filled with the median of numeric columns

**Feature Engineering Code Example:**
```python
df['Avg_Delay_Per_Customer'] = df.groupby('Customer_ID')['Delay_from_due_date'].transform('mean')
df['Max_Delay_Per_Customer'] = df.groupby('Customer_ID')['Delay_from_due_date'].transform('max')

df['EMI_to_Income'] = df['Total_EMI_per_month'] / (df['Monthly_Inhand_Salary'] + 1e-5)
df['Debt_to_Income'] = df['Outstanding_Debt'] / (df['Monthly_Inhand_Salary'] + 1e-5)
df['Utilization_to_Income'] = df['Credit_Utilization_Ratio'] * df['Monthly_Inhand_Salary']

df['Lag1_EMI'] = df.groupby('Customer_ID')['Total_EMI_per_month'].shift(1)
df['Lag1_Delay'] = df.groupby('Customer_ID')['Delay_from_due_date'].shift(1)
df['Lag1_Credit_Util'] = df.groupby('Customer_ID')['Credit_Utilization_Ratio'].shift(1)
for col in ['Lag1_EMI', 'Lag1_Delay', 'Lag1_Credit_Util']:
    df[col].fillna(df[col].median(), inplace=True)
df['EMI_Trend'] = df['Total_EMI_per_month'] - df['Lag1_EMI']
df['Debt_Utilization'] = df['Outstanding_Debt'] * df['Credit_Utilization_Ratio']
df['Loans_per_Card'] = df['Num_of_Loan'] / (df['Num_Credit_Card'] + 1)
df.fillna(df.median(numeric_only=True), inplace=True)
```

**CatBoost Parameters:**
```python
catboost_model = CatBoostClassifier(
    task_type="GPU",
    iterations=4000,
    learning_rate=0.09,
    depth=7,
    cat_features=cat_features_indices,
    random_seed=42,
    eval_metric='MultiClass',
    verbose=100
)
```

**Results:**

Accuracy for CatBoost: **0.7274**

```
Classification Report for CatBoost:
              precision    recall  f1-score   support

        Good       0.70      0.64      0.67      4824
        Poor       0.73      0.75      0.74      7216
    Standard       0.74      0.75      0.74     12960

    accuracy                           0.73     25000
   macro avg       0.72      0.71      0.72     25000
weighted avg       0.73      0.73      0.73     25000
```

---

### Time-Series Feature Engineering: Lag and Rolling Features

To better capture temporal patterns in the data, I engineered lag and rolling window features for each customer:
- **Lag Features:** For key financial columns (e.g., Annual Income, Monthly Inhand Salary, Outstanding Debt, Total EMI per month), I created features representing the value from the previous month for each customer.
- **Rolling Window Features:** For variables like number of delayed payments, interest rate, and number of credit inquiries, I computed the average over the last three months for each customer.
- **Data Sorting and Type Handling:** The data was sorted by customer and month, and all relevant columns were converted to numeric types to ensure correct aggregation.

This approach allowed the model to leverage both recent history and longer-term trends for each customer, improving its ability to predict credit score changes over time.

**RandomForest Results with Lag and Rolling Features:**

```
Classification Report for RandomForest:
              precision    recall  f1-score   support

        Good       0.67      0.61      0.64      4824
        Poor       0.71      0.75      0.73      7216
    Standard       0.73      0.73      0.73     12960

    accuracy                           0.71     25000
   macro avg       0.70      0.70      0.70     25000
weighted avg       0.71      0.71      0.71     25000
```

## 6. Conclusion and Next Steps
This project successfully established a robust pipeline for predicting credit scores from complex, time-series data. The initial RandomForest model achieved a promising accuracy of 70.5% after addressing critical data quality and leakage issues.

**Key Learnings:**
- The importance of identifying the correct data structure (time-series) to prevent data leakage.
- The necessity of a thorough data cleaning and preprocessing phase to handle real-world data imperfections.
- The value of combining model-based feature importance with statistical tests to understand key business drivers.

**Future Improvements:**
- **Advanced Feature Engineering:** The largest potential for improvement lies here. Creating lag features (e.g., previous month's debt) and rolling window features (e.g., 3-month average of delayed payments) would provide the model with crucial historical context.
- **Hyperparameter Tuning:** The baseline models were untuned. Using GridSearchCV or RandomizedSearchCV on the champion RandomForest model could yield significant performance gains.
- **Handling Class Imbalance:** Techniques like SMOTE (Synthetic Minority Over-sampling Technique) could be applied to the training data to improve the model's ability to predict the minority 'Good' class.

---

# Credit Score Prediction & EDA Platform

This project provides a complete pipeline for credit score prediction, exploratory data analysis (EDA), and batch inference using a trained CatBoost model. It includes:

- Data preprocessing and cleaning
- Model training and artifact saving
- Batch prediction via a Streamlit web app
- Automated EDA report generation with visualizations
- Docker support for easy deployment

---

## Project Structure

```
.
├── app/
│   ├── main.py           # Streamlit app for batch prediction
│   ├── logic.py          # Data preprocessing and cleaning logic
│   └── __init__.py
├── data/
│   ├── train.csv         # Training dataset
│   └── test.csv          # Test dataset (for batch prediction)
├── model/
│   ├── catboost_model.pkl    # Trained CatBoost model
│   ├── label_encoder.pkl     # Label encoder for target variable
│   └── X_columns.pkl         # List of feature columns used by the model
├── eda_report/
│   └── *.png             # EDA plots and visualizations
├── run_eda.py            # Script to generate EDA report
├── train_model.py        # Script to train the model
├── requirements.txt      # Python dependencies
├── DockerFile            # Docker build instructions
└── Final_NoteBook.ipynb  # (Original notebook, for reference)
```

---

## Setup

### 1. Local Python Environment

1. **Install Python 3.8+** (with pip)
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### 2. Docker (Recommended)

Build and run the app in a containerized environment:

```bash
# Build the Docker image
docker build -t credit-score-app -f DockerFile .

# Run the Streamlit app
docker run -p 8501:8501 credit-score-app
```

---

## Usage

### 1. Model Training

Before using the app, train the model and generate artifacts:

```bash
python train_model.py
```

Artifacts will be saved in the `model/` directory.

### 2. Exploratory Data Analysis (EDA)

Generate EDA plots from the training data:

```bash
python run_eda.py
```

Plots will be saved in the `eda_report/` directory.

### 3. Batch Prediction Web App

Start the Streamlit app:

```bash
streamlit run app/main.py
```

- Open your browser at [http://localhost:8501](http://localhost:8501)
- Upload a CSV file (e.g., `data/test.csv`)
- View predictions and download results

---

## Features

### Data Preprocessing

- Handles missing values, outliers, and text placeholders
- Cleans and standardizes numeric and categorical features
- Consistent pipeline for both training and inference (see `app/logic.py`)

### Model

- CatBoostClassifier (GPU-accelerated if available)
- Label encoding for target variable
- Artifacts: model, label encoder, feature columns

### EDA

- Automated plots for:
  - Credit Score, Occupation, Credit Mix (countplots)
  - Annual Income, Age, Monthly Inhand Salary, Num Bank Accounts, Num Credit Card, Interest Rate, Delay from Due Date, Num Credit Inquiries (histograms)
  - Payment of Minimum Amount, Payment Behaviour (countplots)
- All plots saved as PNGs in `eda_report/`

### Streamlit App

- Upload CSV for batch prediction
- Data preview, prediction results, and download option
- Uses the same preprocessing as training for consistency

### Docker

- CUDA-enabled base image for GPU support
- All dependencies installed via `requirements.txt`
- Exposes port 8501 for Streamlit

---

## File Descriptions

- `app/main.py`: Streamlit UI for batch predictions
- `app/logic.py`: Data cleaning and preprocessing functions
- `train_model.py`: Trains the CatBoost model and saves artifacts
- `run_eda.py`: Generates EDA plots from training data
- `requirements.txt`: Python dependencies
- `DockerFile`: Docker build instructions

---

## Data

- Place your training and test CSVs in the `data/` directory.
- Example files: `train.csv`, `test.csv`

---

## Outputs

- **Model artifacts:** `model/`
- **EDA plots:** `eda_report/`
- **Prediction results:** Downloadable from the Streamlit app

---

## Notes

- If you update the data or preprocessing logic, retrain the model and regenerate artifacts.
- For GPU acceleration, ensure you have compatible NVIDIA drivers and Docker setup.

---

## License

MIT License (or specify your own)

---

## Contact

For questions or contributions, please open an issue or pull request. 