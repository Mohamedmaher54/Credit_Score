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