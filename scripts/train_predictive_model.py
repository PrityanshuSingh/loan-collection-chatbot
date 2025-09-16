# scripts/train_predictive_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
import joblib
import os
import sys

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_FILE_PATH, PREDICTIVE_MODEL_ASSETS_PATH

warnings.filterwarnings('ignore')

def train_and_save_predictive_model():
    """
    Loads data, trains the XGBoost predictive model, and saves all necessary assets.
    """
    print("--- Starting Predictive Model Training ---")

    # --- Load Data ---
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        print("✓ Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"✗ Error: The file '{DATA_FILE_PATH}' was not found.")
        return

    # --- Data Cleaning and Preprocessing ---
    print("\n--- Data Cleaning and Preprocessing ---")
    categorical_cols = ['Location', 'EmploymentStatus', 'LoanType']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    print("✓ Categorical variables encoded.")

    # --- Feature Engineering ---
    print("\n--- Engineering Features ---")
    df['AgeRiskBucket'] = pd.cut(df['Age'], bins=[0, 25, 40, 60, 100], labels=['<25', '25-40', '40-60', '>60'])
    df['AgeRiskBucket'] = LabelEncoder().fit_transform(df['AgeRiskBucket'])

    r = df['InterestRate'] / (12 * 100)
    n = df['TenureMonths']
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        df['EMI'] = (df['LoanAmount'] * r * (1 + r).pow(n)) / ((1 + r).pow(n) - 1)
    df['DTI'] = df['EMI'] / (df['Income'] / 12)
    df['LTI'] = df['LoanAmount'] / df['Income']
    total_interest = (df['EMI'] * df['TenureMonths']) - df['LoanAmount']
    total_interest = total_interest.clip(lower=0)
    monthly_interest = total_interest / df['TenureMonths']
    df['InterestBurden'] = monthly_interest / (df['Income'] / 12)
    df['TAB'] = df['LoanAmount'] / df['TenureMonths']
    df['MPR'] = df['MissedPayments'] / df['TenureMonths']
    df['AvgDelay'] = df['DelaysDays'] / (df['MissedPayments'] + 1)
    df['PPR'] = df['PartialPayments'] / df['TenureMonths']
    df['BRI'] = (0.5 * df['MPR']) + (0.3 * (df['DelaysDays'] / df['TenureMonths'])) + (0.2 * df['PPR'])
    df['ES'] = 1 / (1 + df['ResponseTimeHours'])
    df['DES'] = (df['AppUsageFrequency'] + (df['WebsiteVisits'] / 30)) / 2
    df['IFI'] = df['InteractionAttempts'] / df['TenureMonths']
    df['SentimentRiskScore'] = (1 - df['SentimentScore']) * df['IFI']
    df['Has_High_Complaints'] = (df['Complaints'] > 2).astype(int)

    high_dti_threshold = df['DTI'].quantile(0.75)
    high_bri_threshold = df['BRI'].quantile(0.75)
    low_des_threshold = df['DES'].quantile(0.25)

    df['Complaints_x_HighDTI'] = df['Has_High_Complaints'] * (df['DTI'] > high_dti_threshold).astype(int)
    df['Complaints_x_HighBRI'] = df['Has_High_Complaints'] * (df['BRI'] > high_bri_threshold).astype(int)
    df['HighBRI_x_LowDES'] = (df['BRI'] > high_bri_threshold).astype(int) * (df['DES'] < low_des_threshold).astype(int)
    df['DTI_x_MissedPayments'] = df['DTI'] * df['MissedPayments']
    df['LoanAmount_x_InterestRate'] = df['LoanAmount'] * df['InterestRate']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df.drop(columns=['Complaints'], inplace=True)
    print("✓ Feature engineering complete.")

    # --- Train-Test Split ---
    X = df.drop(['CustomerID', 'Target'], axis=1)
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- SMOTE ---
    print("\n--- Synthesizing Data using SMOTE ---")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("✓ SMOTE applied to training data.")

    # --- Scaling ---
    print("\n--- Standardizing Data ---")
    scaler = StandardScaler()
    scaler.fit(X_train_resampled)
    X_train = pd.DataFrame(scaler.transform(X_train_resampled), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    print("✓ Data standardized.")

    # --- Hyperparameter Tuning (using a smaller grid for faster execution) ---
    print("\n--- Starting Hyperparameter Tuning ---")
    estimator = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        tree_method='hist'
    )
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'reg_lambda': [1.0, 1.5],
        'colsample_bytree': [0.7, 0.9]
    }
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='roc_auc',
        n_jobs=-1,
        cv=3,
        verbose=1
    )
    grid_search.fit(X_train, y_train_resampled)
    model = grid_search.best_estimator_
    print(f"\n✓ Best parameters found: {grid_search.best_params_}")
    print(f"✓ Best cross-validation ROC AUC score: {grid_search.best_score_:.4f}")

    # --- Final Evaluation ---
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n--- Final Model Evaluation ---")
    print(f"Final ROC AUC Score on Test Set: {roc_auc:.4f}")

    # --- Save Assets ---
    predictive_assets = {
        "model": model,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "high_dti_threshold": high_dti_threshold,
        "high_bri_threshold": high_bri_threshold,
        "low_des_threshold": low_des_threshold,
        "feature_names": list(X.columns)
    }

    os.makedirs(os.path.dirname(PREDICTIVE_MODEL_ASSETS_PATH), exist_ok=True)
    joblib.dump(predictive_assets, PREDICTIVE_MODEL_ASSETS_PATH)
    print(f"\n✓ Predictive model and assets saved to {PREDICTIVE_MODEL_ASSETS_PATH}")
    print("--- Predictive Model Training Complete ---")


if __name__ == '__main__':
    train_and_save_predictive_model()