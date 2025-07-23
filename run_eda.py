import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
DATA_FILE_PATH = 'data/train.csv'
REPORT_DIR = 'eda_report'

def generate_eda_report():
    print("🚀 Starting EDA report generation...")

    os.makedirs(REPORT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_FILE_PATH, low_memory=False)

    plt.style.use('seaborn-v0_8-whitegrid')
    
    plots_to_generate = [
        {'type': 'countplot', 'col': 'Credit_Score', 'title': 'Distribution of Credit Score', 'file': '01_credit_score.png'},
        {'type': 'countplot', 'col': 'Occupation', 'title': 'Distribution of Occupation', 'file': '02_occupation.png', 'rotate_labels': True},
        {'type': 'countplot', 'col': 'Credit_Mix', 'title': 'Distribution of Credit Mix', 'file': '03_credit_mix.png', 'rotate_labels': True},
        {'type': 'histplot', 'col': 'Annual_Income', 'title': 'Distribution of Annual Income', 'file': '04_income.png'},
        {'type': 'histplot', 'col': 'Age', 'title': 'Distribution of Age', 'file': '05_age.png'},
        {'type': 'histplot', 'col': 'Monthly_Inhand_Salary', 'title': 'Distribution of Monthly Inhand Salary', 'file': '06_salary.png'},
    ]

    for plot_info in plots_to_generate:
        plt.figure(figsize=(10, 6))
        
        if plot_info['type'] == 'countplot':
            sns.countplot(x=df[plot_info['col']], palette="mako", order = df[plot_info['col']].value_counts().index)
        elif plot_info['type'] == 'histplot':
            sns.histplot(data=df, x=plot_info['col'], kde=True, bins=30)
            
        plt.title(plot_info['title'], fontsize=14)
        
        if plot_info.get('rotate_labels'):
            plt.xticks(rotation=45, ha='right')
            
        plt.tight_layout()
        save_path = os.path.join(REPORT_DIR, plot_info['file'])
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Saved plot: {save_path}")

    print(f"\n🎉 EDA report generation complete.")

    # --- Additional User-Requested Plots ---
    # 1. Countplot for Payment_of_Min_Amount
    plt.figure(figsize=(10, 6))
    sns.countplot(x=df['Payment_of_Min_Amount'], palette="mako")
    plt.xticks(rotation=45)
    plt.title('Distribution of Payment of Minimum Amount', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, '07_payment_of_min_amount.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved plot: {save_path}")

    # 2. Countplot for Payment_Behaviour
    plt.figure(figsize=(10, 6))
    sns.countplot(x=df['Payment_Behaviour'], palette="mako")
    plt.xticks(rotation=45)
    plt.title('Distribution of Payment Behaviour', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, '08_payment_behaviour.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved plot: {save_path}")

    # 3. Histplot for Monthly_Inhand_Salary (again, as requested)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Monthly_Inhand_Salary', kde=True)
    plt.title('Distribution of Monthly Inhand Salary (User Requested)', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, '09_monthly_inhand_salary_user.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved plot: {save_path}")

    # 4. Histplot for Num_Bank_Accounts
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Num_Bank_Accounts', kde=True)
    plt.title('Distribution of Num_Bank_Accounts', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, '10_num_bank_accounts.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved plot: {save_path}")

    # 5. Histplot for Num_Credit_Card
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Num_Credit_Card', kde=True)
    plt.title('Distribution of Num_Credit_Card', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, '11_num_credit_card.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved plot: {save_path}")

    # 6. Histplot for Interest_Rate
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Interest_Rate', kde=True)
    plt.title('Distribution of Interest_Rate', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, '12_interest_rate.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved plot: {save_path}")

    # 7. Histplot for Delay_from_due_date
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Delay_from_due_date', kde=True)
    plt.title('Distribution of Delay_from_due_date', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, '13_delay_from_due_date.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved plot: {save_path}")

    # 8. Histplot for Num_Credit_Inquiries
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Num_Credit_Inquiries', kde=True)
    plt.title('Distribution of Num_Credit_Inquiries', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, '14_num_credit_inquiries.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved plot: {save_path}")

    # 9. Histplot for Age (user requested)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Age', kde=True)
    plt.title('Distribution of AGE', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, '15_age_user.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved plot: {save_path}")

if __name__ == '__main__':
    generate_eda_report()