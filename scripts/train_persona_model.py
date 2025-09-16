# scripts/train_persona_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
import joblib
import os
import sys

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_FILE_PATH, PERSONA_MODEL_ASSETS_PATH

def train_and_save_persona_model():
    """
    Loads data, trains the KMeans persona clustering model, and saves all necessary assets.
    """
    print("--- Starting Persona Model Training ---")
    OPTIMAL_K = 5

    # --- Load Data and Filter Defaulters ---
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        df_defaulters = df[df['Target'] == 1].copy()
        print(f"✓ Dataset loaded. Found {len(df_defaulters)} defaulters.")
    except FileNotFoundError:
        print(f"✗ Error: The file '{DATA_FILE_PATH}' was not found.")
        return

    df_analysis = df_defaulters.copy()

    # --- Feature Engineering ---
    print("\n--- Engineering Behavioral Features ---")
    df_analysis['DebtToIncomeRatio'] = (df_analysis['LoanAmount'] / (df_analysis['Income'] + 1)).clip(upper=5)
    df_analysis['MonthlyPaymentBurden'] = (df_analysis['LoanAmount'] * (1 + df_analysis['InterestRate']/100)) / df_analysis['TenureMonths']
    df_analysis['PaymentToIncomeRatio'] = df_analysis['MonthlyPaymentBurden'] / (df_analysis['Income']/12 + 1)
    median_interest = df_analysis['InterestRate'].median()
    df_analysis['InterestPremium'] = df_analysis['InterestRate'] - median_interest
    df_analysis['PaymentConsistency'] = 1 / (1 + df_analysis['MissedPayments'])
    df_analysis['DelayIntensity'] = df_analysis['DelaysDays'] / (df_analysis['TenureMonths'] * 30 + 1)
    df_analysis['PartialPaymentRate'] = df_analysis['PartialPayments'] / (df_analysis['MissedPayments'] + 1)
    df_analysis['ResponseRate'] = 1 / (1 + df_analysis['ResponseTimeHours']/24)
    df_analysis['InteractionIntensity'] = df_analysis['InteractionAttempts'] / (df_analysis['TenureMonths'] + 1)
    df_analysis['DigitalEngagement'] = (df_analysis['AppUsageFrequency'] + df_analysis['WebsiteVisits']/30) / 2
    df_analysis['SentimentNormalized'] = (df_analysis['SentimentScore'] + 1) / 2
    df_analysis['ComplaintRate'] = df_analysis['Complaints'] / (df_analysis['InteractionAttempts'] + 1)
    employment_risk = {'Unemployed': 1.0, 'Student': 0.7, 'Self-Employed': 0.5, 'Salaried': 0.2}
    df_analysis['EmploymentRisk'] = df_analysis['EmploymentStatus'].map(employment_risk).fillna(0.5)
    df_analysis['AgeRisk'] = np.where(df_analysis['Age'] < 25, 0.8, np.where(df_analysis['Age'] > 60, 0.7, 0.4))
    df_analysis['EarlyDefaultRisk'] = np.where(df_analysis['TenureMonths'] < 6, 1.0, np.where(df_analysis['TenureMonths'] < 12, 0.7, 0.4))
    print("✓ Behavioral features created.")

    # --- Composite Scores ---
    print("\n--- Creating Composite Scores ---")
    df_analysis['FinancialCapacity'] = ((1-df_analysis['DebtToIncomeRatio'])*0.4 + (1-df_analysis['PaymentToIncomeRatio'])*0.3 + (1-df_analysis['InterestPremium'])*0.15 + (1-df_analysis['EmploymentRisk'])*0.15)
    df_analysis['PaymentReliability'] = (df_analysis['PaymentConsistency']*0.4 + (1-df_analysis['DelayIntensity'])*0.3 + df_analysis['PartialPaymentRate']*0.2 + (1-df_analysis['EarlyDefaultRisk'])*0.1)
    df_analysis['EngagementQuality'] = (df_analysis['ResponseRate']*0.3 + df_analysis['InteractionIntensity']*0.2 + df_analysis['DigitalEngagement']*0.3 + df_analysis['SentimentNormalized']*0.2)
    df_analysis['CooperationLevel'] = (df_analysis['SentimentNormalized']*0.4 + (1-df_analysis['ComplaintRate'])*0.3 + df_analysis['ResponseRate']*0.3)
    df_analysis['OverallRiskScore'] = ((1-df_analysis['FinancialCapacity'])*0.35 + (1-df_analysis['PaymentReliability'])*0.35 + (1-df_analysis['EngagementQuality'])*0.15 + (1-df_analysis['CooperationLevel'])*0.15)
    
    clustering_features = ['FinancialCapacity', 'PaymentReliability', 'EngagementQuality', 'CooperationLevel', 'OverallRiskScore']
    features_for_scoring = ['DebtToIncomeRatio', 'PaymentToIncomeRatio', 'InterestPremium', 'PaymentConsistency', 'DelayIntensity', 'PartialPaymentRate', 'ResponseRate', 'InteractionIntensity', 'DigitalEngagement', 'SentimentNormalized', 'ComplaintRate', 'EmploymentRisk', 'AgeRisk', 'EarlyDefaultRisk']

    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df_analysis[features_for_scoring]), columns=features_for_scoring)
    
    print("✓ Composite scores calculated.")

    # --- Outlier Detection ---
    print("\n--- Detecting and Removing Outliers ---")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outlier_labels = iso_forest.fit_predict(df_analysis[clustering_features].fillna(df_analysis[clustering_features].median()))
    df_clean = df_analysis[outlier_labels == 1].copy()
    print(f"✓ Removed {len(df_analysis) - len(df_clean)} outliers.")

    # --- Clustering ---
    print("\n--- Performing Final Clustering ---")
    X_clustering = df_clean[clustering_features].fillna(df_clean[clustering_features].median())
    X_scaled = StandardScaler().fit_transform(X_clustering)
    kmeans_final = KMeans(n_clusters=OPTIMAL_K, init='k-means++', n_init=50, random_state=42)
    df_clean['Cluster'] = kmeans_final.fit_predict(X_scaled)
    print(f"✓ Clustering complete with K={OPTIMAL_K}.")
    
    # --- Persona Mapping ---
    cluster_stats = df_clean.groupby('Cluster')[clustering_features].mean().sort_values('OverallRiskScore')
    risk_order = cluster_stats.index.tolist()
    cluster_mapping = {old: new for new, old in enumerate(risk_order)}
    print("✓ Persona mapping created based on risk score.")

    # --- Define Personas ---
    persona_definitions = {
        0: {'name': 'Cooperative Optimizers', 'risk_level': 'Low'},
        1: {'name': 'Willing but Struggling', 'risk_level': 'Medium-Low'},
        2: {'name': 'Passive Defaulters', 'risk_level': 'Medium'},
        3: {'name': 'Stressed Resistors', 'risk_level': 'Medium-High'},
        4: {'name': 'High-Risk Avoiders', 'risk_level': 'High'}
    }

    # --- Save Assets ---
    persona_assets = {
        "model": kmeans_final,
        "training_data_for_scaler": X_clustering,
        "persona_scaler": StandardScaler().fit(X_clustering),
        "cluster_mapping": cluster_mapping,
        "definitions": persona_definitions,
        "employment_risk_map": employment_risk,
        "median_interest": median_interest,
        "imputer_data_median": df_imputed.median(),
        "persona_feature_names": clustering_features
    }

    os.makedirs(os.path.dirname(PERSONA_MODEL_ASSETS_PATH), exist_ok=True)
    joblib.dump(persona_assets, PERSONA_MODEL_ASSETS_PATH)
    print(f"\n✓ Persona model and assets saved to {PERSONA_MODEL_ASSETS_PATH}")
    print("--- Persona Model Training Complete ---")

if __name__ == '__main__':
    train_and_save_persona_model()