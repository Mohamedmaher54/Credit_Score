# logic.py

import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    """
    Applies a series of cleaning and preprocessing steps to a DataFrame.
    This pipeline is designed to work for single inputs from the app.
    """
    df_processed = df.copy()

    # --- Clean Underscore Formatting ---
    df_processed = clean_underscore_formatting(df_processed)

    # --- Clean Text Placeholders ---
    df_processed = clean_text_placeholders(df_processed)

    # --- Fix Age Outliers ---
    df_processed = fix_age_outliers(df_processed)

    # --- Fix Outliers Per Customer ---
    df_processed = fix_outliers_per_customer(df_processed)

    # --- Fix Delayed Payments ---
    df_processed = fix_delayed_payments(df_processed)

    # --- Impute Occupation By Customer ---
    df_processed = impute_occupation_by_customer(df_processed)

    # --- Clean Categorical Features ---
    df_processed = clean_categorical_features(df_processed)

    # --- Impute Payment Behaviour ---
    df_processed = impute_payment_behaviour(df_processed)

    # --- Convert Credit History Age ---
    df_processed = convert_credit_history_age(df_processed)

    return df_processed


def clean_underscore_formatting(df):
    """
    Clean underscore formatting issues like __1000__ or 809_.
    
    Args:
        df (pd.DataFrame): DataFrame with columns that need cleaning.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns.
    """
    print("Cleaning underscore formatting...")

    # Define columns that might have underscore issues
    numeric_cols_with_underscores = [
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
        'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
        'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
        'Amount_invested_monthly', 'Monthly_Balance'
    ]

    for col in numeric_cols_with_underscores:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('_', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def clean_text_placeholders(df):
    """
    Clean specific text placeholders like '_______' and '!@9#%8' by converting them to NaN.
    
    Args:
        df (pd.DataFrame): DataFrame with categorical columns to clean.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned text columns.
    """
    print("Cleaning text placeholders...")
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].replace('_______', np.nan)
        df[col] = df[col].replace('!@9#%8', np.nan)
    
    return df


def fix_age_outliers(df):
    """
    Fix age outliers and ensure consistency per customer.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Age' column.
    
    Returns:
        pd.DataFrame: DataFrame with corrected age values.
    """
    print("Fixing age outliers...")

    def fix_customer_age(group):
        ages = pd.to_numeric(group['Age'], errors='coerce')

        # Remove outliers (age should be reasonable)
        valid_ages = ages[(ages >= 18) & (ages <= 100)]
        if len(valid_ages) > 0:
            most_frequent_age = valid_ages.mode().iloc[0] if len(valid_ages.mode()) > 0 else valid_ages.median()
            group['Age'] = most_frequent_age
        else:
            all_ages = pd.to_numeric(df['Age'], errors='coerce')
            overall_median = all_ages[(all_ages >= 18) & (all_ages <= 100)].median()
            group['Age'] = overall_median if pd.notna(overall_median) else 30
        return group

    df = df.groupby('Customer_ID').apply(fix_customer_age).reset_index(drop=True)
    return df


def fix_outliers_per_customer(df):
    """
    Fix outliers for columns that should be constant per customer.
    
    Args:
        df (pd.DataFrame): DataFrame containing customer columns to fix.
    
    Returns:
        pd.DataFrame: DataFrame with outliers fixed per customer.
    """
    print("Fixing outliers for customer-consistent columns...")

    customer_constant_cols = [
        'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Num_Credit_Inquiries',
        'Total_EMI_per_month', 'Num_of_Delayed_Payment'
    ]

    def fix_customer_outliers(group, col):
        values = pd.to_numeric(group[col], errors='coerce').dropna()
        if len(values) > 0:
            try:
                most_frequent = values.mode().iloc[0] if len(values.mode()) > 0 else values.median()
            except:
                most_frequent = values.median()
            group[col] = most_frequent
        return group

    for col in customer_constant_cols:
        if col in df.columns:
            df = df.groupby('Customer_ID').apply(lambda x: fix_customer_outliers(x, col)).reset_index(drop=True)

    return df


def fix_delayed_payments(df):
    """
    Fix Num_of_Delayed_Payment: make positive, then fix outliers.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Num_of_Delayed_Payment' and 'Delay_from_due_date'.
    
    Returns:
        pd.DataFrame: DataFrame with fixed negative values and outliers.
    """
    print("Fixing Negative values in Delayed payment and num of delayed payment...")

    if 'Num_of_Delayed_Payment' and 'Delay_from_due_date' in df.columns:
        df['Num_of_Delayed_Payment'] = pd.to_numeric(df['Num_of_Delayed_Payment'], errors='coerce').abs()
        df['Delay_from_due_date'] = pd.to_numeric(df['Delay_from_due_date'], errors='coerce').abs()

    return df


def impute_occupation_by_customer(df):
    """
    Fills missing 'Occupation' values with the most frequent occupation for each customer.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Occupation' column.
    
    Returns:
        pd.DataFrame: DataFrame with imputed 'Occupation'.
    """
    print("Imputing 'Occupation' based on customer's most frequent value...")

    if 'Occupation' in df.columns:
        df.loc[df['Occupation'] == '_______', 'Occupation'] = np.nan
        df['Occupation'] = df.groupby('Customer_ID')['Occupation'].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown")
        )
    
    return df


def clean_categorical_features(df):
    """
    Cleans trailing underscores from specific categorical columns like Credit_Mix.
    
    Args:
        df (pd.DataFrame): DataFrame with categorical columns.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned categorical columns.
    """
    print("Cleaning categorical features like 'Credit_Mix'...")

    categorical_clean_cols = ['Credit_Mix']

    for col in categorical_clean_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('_', '', regex=False)

    return df


def impute_payment_behaviour(df):
    """
    Fills missing 'Payment_Behaviour' values with the most frequent occupation for each customer.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Payment_Behaviour' column.
    
    Returns:
        pd.DataFrame: DataFrame with imputed 'Payment_Behaviour'.
    """
    print("Imputing 'Payment_Behaviour' with most frequent value...")

    if 'Payment_Behaviour' in df.columns:
        df.loc[df['Payment_Behaviour'] == '!@9#%8', 'Payment_Behaviour'] = np.nan
        
        imputer = SimpleImputer(strategy="most_frequent")
        df[['Payment_Behaviour']] = imputer.fit_transform(df[['Payment_Behaviour']])
    
    return df


def convert_credit_history_age(df):
    """
    Convert Credit_History_Age to total months.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Credit_History_Age' column.
    
    Returns:
        pd.DataFrame: DataFrame with 'Credit_History_Age' converted to months.
    """
    print("Converting credit history age to months...")

    def parse_credit_history(text):
        if pd.isna(text) or text == 'NA':
            return np.nan
        text = str(text)
        years_match = re.search(r'(\d+)\s*Years?', text, re.IGNORECASE)
        months_match = re.search(r'(\d+)\s*Months?', text, re.IGNORECASE)
        years = int(years_match.group(1)) if years_match else 0
        months = int(months_match.group(1)) if months_match else 0
        return (years * 12) + months

    df['Credit_History_Age_Months'] = df['Credit_History_Age'].apply(parse_credit_history)

    def fill_credit_history(group):
        if group['Credit_History_Age_Months'].isna().all():
            return group
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        group['Month_Num'] = group['Month'].map({month: i for i, month in enumerate(month_order)})
        group = group.sort_values('Month_Num')

        first_valid = group['Credit_History_Age_Months'].first_valid_index()
        if first_valid is not None:
            base_value = group.loc[first_valid, 'Credit_History_Age_Months']
            for i, idx in enumerate(group.index):
                if pd.isna(group.loc[idx, 'Credit_History_Age_Months']):
                    group.loc[idx, 'Credit_History_Age_Months'] = base_value + i

        group = group.drop('Month_Num', axis=1)
        return group

    df = df.groupby('Customer_ID').apply(fill_credit_history).reset_index(drop=True)
    return df


# --- Main Block for Testing ---
if __name__ == "__main__":
    # Example: Load your dataset here
    # df = pd.read_csv('your_file.csv')

    # For demonstration purposes, we'll load an example dataframe:


    # Apply cleaning functions
    df_cleaned = preprocess_data(df)

    # Output cleaned dataframe
    print("\nCleaned DataFrame:")
    print(df_cleaned)
